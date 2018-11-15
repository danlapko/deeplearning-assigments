import logging
import os

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import save_image


class VAETrainer:

    def __init__(self, model, train_loader, test_loader, optimizer,
                 loss_function, log_dir, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, epoch, log_interval):
        self.model.train()
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            # TODO >>> your code here

            data = data.to(self.device)
            self.optimizer.zero_grad()

            data_reconstr, mu, logvar = self.model(data)
            train_loss = self.loss_function(data_reconstr, data, mu, logvar)
            train_loss.backward()

            # TODO <<< your code here

            epoch_loss += train_loss
            norm_train_loss = train_loss / len(data)

            self.optimizer.step()
            if batch_idx % log_interval == 0:
                msg = f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ' \
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t' \
                      f'Loss: {norm_train_loss:.6f}'
                logging.info(msg)

                batch_size = self.train_loader.batch_size
                train_size = len(self.train_loader.dataset)
                batches_per_epoch_train = train_size // batch_size
                self.writer.add_scalar(tag='data/train_loss',
                                       scalar_value=norm_train_loss,
                                       global_step=batches_per_epoch_train * epoch + batch_idx)

        epoch_loss /= len(self.train_loader.dataset)
        logging.info(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar(tag='data/train_epoch_loss',
                               scalar_value=epoch_loss,
                               global_step=epoch)

    def test(self, epoch, batch_size, log_interval):
        self.model.eval()
        test_epoch_loss = 0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.test_loader):
                # TODO >>> your code here

                data = data.to(self.device)
                data_reconstr, mu, logvar = self.model(data)
                test_loss = self.loss_function(data_reconstr, data, mu, logvar)

                # TODO <<< your code here

                test_epoch_loss += test_loss

                if batch_idx % log_interval == 0:
                    msg = 'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data),
                        len(self.test_loader.dataset),
                               100. * batch_idx / len(self.test_loader),
                               test_loss / len(data))
                    logging.info(msg)

                    batches_per_epoch_test = len(self.test_loader.dataset) // batch_size
                    self.writer.add_scalar(tag='data/test_loss',
                                           scalar_value=test_loss / len(data),
                                           global_step=batches_per_epoch_test * (epoch - 1) + batch_idx)

            test_epoch_loss /= len(self.test_loader.dataset)
            logging.info('====> Test set loss: {:.4f}'.format(test_epoch_loss))
            self.writer.add_scalar(tag='data/test_epoch_loss',
                                   scalar_value=test_epoch_loss,
                                   global_step=epoch)
        self.plot_generated(epoch, batch_size)

    def plot_generated(self, epoch, batch_size):
        # TODO >>> your code here

        with torch.no_grad():
            sample, _ = next(iter(self.test_loader))
            sample = sample[:32]
            z = self.model.embed(sample.to(self.device))
            sample_reconstr = self.model.decode(z).cpu()
            sample_reconstr = sample_reconstr.view(32, 1, 28, 28)

            out = torch.cat((sample, sample_reconstr), 0)
            save_image(out,
                       'results_vae/epoch_' + str(epoch) + '.png')

        # TODO <<< your code here

    def save(self, checkpoint_path):
        dir_name = os.path.dirname(checkpoint_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
