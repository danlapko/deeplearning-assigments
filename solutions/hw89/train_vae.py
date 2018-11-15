# tensorboard --logdir='logs_xboard_vae'
import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torchvision import transforms

from solutions.hw89.vae.vae import VAE, loss_function
from solutions.hw89.vae.vaetrainer import VAETrainer

device = torch.device("cuda")


def get_config():
    parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

    parser.add_argument('--log-root', type=str, default='logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=28,
                        help='size of images to generate')
    parser.add_argument('--log_metrics_every', type=int, default=10)
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    # TODO your code here
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(config.data_root, train=True, download=True,
                              transform=transforms.ToTensor()),
        batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(config.data_root, train=False, transform=transforms.ToTensor()),
        batch_size=config.batch_size, shuffle=True)

    vae_model = VAE(config.image_size).to(device)

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
    trainer = VAETrainer(vae_model, train_loader, test_loader, optimizer, loss_function, "logs_xboard_vae",
                         device=device)

    for epoch in range(config.epochs):
        trainer.train(epoch + 1, config.log_metrics_every)
        trainer.test(epoch + 1, config.batch_size, config.log_metrics_every)


if __name__ == '__main__':
    main()
