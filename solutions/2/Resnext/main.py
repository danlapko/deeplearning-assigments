import random
import torch

from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from solutions.two.Resnext.logger import Logger

from solutions.two.Resnext.resnext import resnext50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_random_params():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True


step = 0


def train(epoch_num, model, optimizer, loss_func, train_loader, logger):
    model.train()
    global step
    for batch_idx, (x, y) in enumerate(train_loader):
        step += 1
        x, y = x.to(device), y.to(device)

        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()

        y_ = model(x)

        loss = loss_func(y_, y)
        loss.backward()
        optimizer.step()

        pred = y_.data.max(1)[1]
        correct = pred.eq(y.data).cpu().sum()

        acc = float(correct) / len(pred)

        if batch_idx % 100 == 0:
            # Tensorboard Logging

            logger.scalar_summary('loss', loss.item(), step + 1)
            logger.scalar_summary('accuracy', acc, step + 1)
            logger.image_summary(y[:10].cpu().numpy(), x[:10].cpu().numpy(), step + 1)


def main():
    weights_path = "./weigts/resnext50.pth"

    fix_random_params()

    model = resnext50(num_classes=10, weigts_path=None).to(device)
    logger = Logger('./logs')

    n_epoches = 5

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_loader = DataLoader(
        datasets.SVHN(root="./data", split="train", download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                      ])),
        batch_size=64, shuffle=True, num_workers=4)

    for i_epoch in range(n_epoches):
        train(i_epoch, model, optimizer, loss_func, train_loader, logger)

    torch.save(model.state_dict(), weights_path)


if __name__ == "__main__":
    main()
