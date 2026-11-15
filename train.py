"""Step 1: train ResNet-20 on CIFAR-10 (about 10-15 min on an RTX 4060 Ti) and save it."""
import torch, torch.nn as nn, time
from common import ResNet20, loaders, accuracy, DEV

EPOCHS = 40


def main():
    torch.manual_seed(0)
    model = ResNet20().to(DEV)
    train, test = loaders()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, epochs=EPOCHS, steps_per_epoch=len(train))
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = n = 0
        for x, y in train:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * y.numel(); n += y.numel()
        msg = f"epoch {ep+1:2d}/{EPOCHS}  train loss {tot/n:.3f}  ({time.time()-t0:.0f}s)"
        if ep % 5 == 4 or ep == EPOCHS - 1:
            msg += f"  test acc {accuracy(model, test):.2f}%"
        print(msg, flush=True)
    torch.save(model.state_dict(), "resnet20.pt")
    print("saved resnet20.pt")


if __name__ == "__main__":   # required on Windows: DataLoader workers re-import this file
    main()
