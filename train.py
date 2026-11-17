"""Step 1: train ResNet-20 on CIFAR-10 (about 10-15 min on an RTX 4060 Ti) and save it.

    python train.py                 -> resnet20.pt
    python train.py --cluster 5e-4  -> resnet20_pc.pt, trained with the piece-wise
                                       clustering penalty (He et al., CVPR 2020)
"""
import argparse, torch, torch.nn as nn, time
from common import ResNet20, loaders, accuracy, quant_layers, DEV

EPOCHS = 40


def cluster_penalty(model):
    """Pull the positive weights of each layer toward their mean, and the negative
    weights toward theirs. A layer then looks like two tight clusters at +a and -b,
    so one flipped bit moves a weight further outside where the rest of the layer sits."""
    p = 0.0
    for _, m in quant_layers(model):
        w = m.weight
        pos, neg = w[w > 0], w[w < 0]
        if pos.numel(): p = p + ((pos - pos.mean()) ** 2).sum()
        if neg.numel(): p = p + ((neg - neg.mean()) ** 2).sum()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", type=float, default=0.0, help="clustering penalty weight (0 = off)")
    args = ap.parse_args()
    out = "resnet20_pc.pt" if args.cluster else "resnet20.pt"

    torch.manual_seed(0)
    model = ResNet20().to(DEV)
    train, test = loaders()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, epochs=EPOCHS, steps_per_epoch=len(train))
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = pen = n = 0
        for x, y in train:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                loss = lossf(model(x), y)
            total = loss
            if args.cluster:                       # penalty in fp32, outside autocast
                cp = args.cluster * cluster_penalty(model)
                total = loss + cp; pen += cp.item() * y.numel()
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * y.numel(); n += y.numel()
        msg = f"epoch {ep+1:2d}/{EPOCHS}  train loss {tot/n:.3f}"
        if args.cluster: msg += f"  penalty {pen/n:.3f}"
        msg += f"  ({time.time()-t0:.0f}s)"
        if ep % 5 == 4 or ep == EPOCHS - 1:
            msg += f"  test acc {accuracy(model, test):.2f}%"
        print(msg, flush=True)
    torch.save(model.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":   # required on Windows: DataLoader workers re-import this file
    main()
