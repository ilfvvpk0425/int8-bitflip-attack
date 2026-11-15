"""Shared pieces: ResNet-20 for CIFAR-10, data loaders, evaluation, int8 weight quantization."""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

# ---------------- model ----------------
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))

class ResNet20(nn.Module):
    def __init__(self, n=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.bn = nn.BatchNorm2d(16)
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                layers.append(Block(cin, cout, stride if i == 0 else 1)); cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

# ---------------- data ----------------
def loaders(batch=256, workers=2):
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    te = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=te)
    return (torch.utils.data.DataLoader(train, batch, shuffle=True, num_workers=workers, pin_memory=True),
            torch.utils.data.DataLoader(test, 1000, shuffle=False, num_workers=workers, pin_memory=True))

@torch.no_grad()
def accuracy(model, loader, max_batches=None):
    model.eval(); ok = n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches: break
        x, y = x.to(DEV), y.to(DEV)
        ok += (model(x).argmax(1) == y).sum().item(); n += y.numel()
    return 100.0 * ok / n

# ---------------- int8 quantization ----------------
def quant_layers(model):
    """Conv and Linear layers whose weights we store as int8 (the 'memory' an attacker can flip)."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

def quantize(model):
    """Symmetric per-layer int8. Returns {name: (int8 tensor, scale)} and loads the
    de-quantized weights into the model so it runs exactly like the 8-bit version."""
    q = {}
    for name, m in quant_layers(model):
        w = m.weight.data
        scale = w.abs().max() / 127.0
        qi = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
        q[name] = (qi, scale)
        m.weight.data = qi.float() * scale
    return q

def load_q(model, q):
    mods = dict(model.named_modules())
    for name, (qi, scale) in q.items():
        mods[name].weight.data = qi.float() * scale

def flip_bits_global(q, names, bit_ids):
    """Flip many bits at once. bit_ids index the concatenation of all layers' bits
    (weight i of the concatenation owns bits 8*i .. 8*i+7, bit 0 = LSB)."""
    sizes = torch.tensor([q[n][0].numel() for n in names])
    starts = torch.cumsum(sizes, 0) - sizes
    w_ids, bits = bit_ids // 8, bit_ids % 8
    layer = torch.searchsorted(torch.cumsum(sizes, 0), w_ids, right=True)
    for li, name in enumerate(names):
        sel = layer == li
        if not sel.any(): continue
        qi = q[name][0]
        u = qi.view(-1).cpu().view(torch.uint8).clone()
        idx = w_ids[sel] - starts[li]
        masks = (1 << bits[sel]).to(torch.uint8)
        # a weight may receive several flips (different bits): apply them one at a time
        for i, m in zip(idx.tolist(), masks.tolist()):
            u[i] ^= m
        q[name] = (u.view(torch.int8).view_as(qi).to(qi.device), q[name][1])

def flip_bit(qi_flat, idx, bit):
    """Flip one bit (0 = LSB, 7 = sign/MSB) of a two's-complement int8 value, in place."""
    v = int(qi_flat[idx].item()) & 0xFF
    v ^= (1 << bit)
    qi_flat[idx] = v - 256 if v >= 128 else v
