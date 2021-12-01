from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union

from torch.cuda.amp import autocast, GradScaler

from data.dataset import SRDatasets
from metrics.metrics import DefaultMetrics
from models.abc_discriminator import Discriminator
from models.abc_generator import Generator
from losses.abc_loss import CustomLoss
from datetime import datetime
from torch.optim import Optimizer
import torch
from torch import device as TorchDevice
from torch import Tensor
import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logging

from utils import visualize_img_from_array
from utils import in_ipynb
from collections import defaultdict
from torch.optim.lr_scheduler import \
    LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CyclicLR, CosineAnnealingWarmRestarts
from training import ROOT_DIR

if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

TEXT_MSG_PER_EVERY_N_STEP = '''
mode: {mode}, {epoch} epoch, {step} batch; 
    generative losses: {gen_loss:.5f}, 
    discriminator losses: {disc_loss}\n
'''

# todo add info about error in logs
# todo add saving transforms in pipeline
# todo change weights of gen_losses in train  like while its only start
#  we need to generate only similar pixels and then we can add some other things
# todo add saving results from visualize

@dataclass
class Trainer:
    generator: Generator
    generator_optimizer: Optimizer
    generator_loss: CustomLoss
    config: dict

    datasets: SRDatasets
    metrics: DefaultMetrics

    log_dir: str = f'logs_dir_about_train_{str(datetime.now())}'
    log_filename: str = f'logs_{str(datetime.now())}.txt'
    full_log_path: str = None

    device: TorchDevice = torch.device("cuda:0")

    discriminator: Optional[Discriminator] = None
    discriminator_optimizer: Optional[Optimizer] = None
    discriminator_loss: Optional[CustomLoss] = None
    discriminator_scheduler: Optional[
        Union[LambdaLR, StepLR, MultiStepLR, CosineAnnealingWarmRestarts, ExponentialLR, CosineAnnealingLR, CyclicLR]
    ] = None

    generator_scheduler: Optional[
        Union[LambdaLR, StepLR, MultiStepLR, CosineAnnealingWarmRestarts, ExponentialLR, CosineAnnealingLR, CyclicLR]
    ] = None

    logger = logging.getLogger(__name__)
    train_discriminator_every_n_step: int = 5
    verbose_every_n_steps: int = 150
    getting_average_by_last_n: int = 20
    current_epoch: int = 0
    show_predicted_img_every_n_batch: int = 250

    def __post_init__(self):
        self.gen_scaler = GradScaler()
        self.disc_scaler = GradScaler()
        self.full_log_path = os.path.join(ROOT_DIR, self.log_dir, self.log_filename)

        if self.log_dir is None:
            self.log_dir = f'logs_dir_about_train_{str(datetime.now())}'

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.config_logger()
        self.check_all_asserts()

    def config_logger(self):
        f_handler = logging.FileHandler(self.full_log_path)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(f_format)
        self.logger.addHandler(f_handler)
        self.logger.setLevel(logging.INFO)

    def check_all_asserts(self):
        self.assert_everything_fine_with_memory_and_sizes()
        self.assert_datasets_is_right()
        self.assert_networks_fine()
        self.logger.info("Every asserts of trainer are fine!")

    def assert_networks_fine(self):
        assert self.generator is not None
        assert self.generator_optimizer is not None
        assert self.generator_loss is not None
        if self.discriminator is not None:
            assert self.discriminator_optimizer is not None
            assert self.discriminator_loss is not None

    def assert_everything_fine_with_memory_and_sizes(self):
        batch = self.datasets.train_loader.dataset[0]
        lr_img = batch['lr_img']
        lr_img = lr_img.reshape(1, *lr_img.shape)
        sr_img = batch['sr_img']
        sr_img = sr_img.reshape(1, *sr_img.shape)
        pred_sr_img = self.generator(lr_img)
        assert pred_sr_img.shape == sr_img.shape, f'{pred_sr_img.shape} and {sr_img.shape}'
        if self.discriminator is not None:
            pred_disc, real_disc = self.discriminator(pred_sr_img, sr_img)
            assert pred_disc.shape == real_disc.shape and (pred_disc <= 1.).all() and (real_disc <= 1.).all()

    def assert_datasets_is_right(self):
        assert self.datasets.train_loader is not None
        assert self.datasets.train_dataset is not None
        assert self.datasets.val_loader is not None
        assert self.datasets.val_dataset is not None
        assert self.datasets.test_dataset is not None
        assert self.datasets.test_loader is not None

    def train(self, epochs: int, starting_epoch: int = 0):
        losses = {
            'train_gen_loss': [],
            'train_disc_loss': [],
            'val_gen_loss': [],
            'val_disc_loss': [],
            'epoch': [],
        }

        for epoch in tqdm(range(starting_epoch, epochs)):
            self.current_epoch = epoch

            train_gen_loss, train_disc_loss = self._train1epoch()

            val_gen_loss, val_disc_loss = self._val()

            losses['train_gen_loss'].append(train_gen_loss)
            losses['train_disc_loss'].append(train_disc_loss)
            losses['val_gen_loss'].append(val_gen_loss)
            losses['val_disc_loss'].append(val_disc_loss)
            losses['epoch'].append(epoch)
            self.on_epoch_end(losses)

        test_gen_loss, test_disc_loss = self._test()
        self.logger.info(f'final losses: test_gen_loss={test_gen_loss:.5f}, test_disc_loss={test_disc_loss}')

    def _step(self, data, calculate_metrics: bool = False) -> Tuple[Tensor, Optional[Tensor], Optional[Dict]]:
        lr_img, sr_img = data['lr_img'].to(self.device), data['sr_img'].to(self.device)
        metrics = None
        discriminator_loss_on_step = None
        with autocast():
            pred_sr_img = self.generator(lr_img)
            cloned_pred_sr_img = pred_sr_img.clone().detach()
            if self.discriminator:
                pred_disc_val, disc_val = self.discriminator(cloned_pred_sr_img, sr_img)
                discriminator_loss_on_step = self.discriminator_loss(pred_disc_val, disc_val)

            generator_loss_on_step = self.generator_loss(
                pred_sr_img, sr_img, pred_disc_val.clone().detach(), disc_val.clone().detach()
            )

            cloned_sr_img = sr_img.clone().detach()
            cloned_lr_img = lr_img.clone().detach()
            cloned_gen_loss = generator_loss_on_step.clone().detach()
            cloned_disc_loss = discriminator_loss_on_step.clone().detach() \
                if discriminator_loss_on_step is not None else None

            if calculate_metrics:
                metrics = self.metrics.calculate_metrics(pred_imgs=cloned_pred_sr_img, real_imgs=cloned_sr_img)

        if self.show_predicted_img_every_n_batch and \
                np.random.randint(1, self.show_predicted_img_every_n_batch + 2, 1)[0]\
                    >= self.show_predicted_img_every_n_batch:
            self.visualize(cloned_pred_sr_img, cloned_lr_img, cloned_sr_img, cloned_gen_loss.item(),
                           cloned_disc_loss.item() if discriminator_loss_on_step is not None else None)

        return generator_loss_on_step, discriminator_loss_on_step, metrics

    def _train1epoch(self) -> Tuple[float, float]:
        self.change_training_mode(train=True)

        losses = {
            'gen_loss': [],
            'disc_loss': []
        }
        disc_losses = []

        for step, data in enumerate(tqdm(self.datasets.train_loader)):
            self.generator_optimizer.zero_grad()
            if self.discriminator:
                self.discriminator_optimizer.zero_grad()

            gen_loss, disc_loss, _ = self._step(data)
            disc_losses.append(disc_loss)

            self.gen_scaler.scale(gen_loss).backward()
            self.gen_scaler.step(self.generator_optimizer)
            losses['gen_loss'].append(gen_loss.item())
            if self.discriminator is not None and step % self.train_discriminator_every_n_step == 0:
                assert disc_loss is not None, 'disc_loss is none'
                self.disc_scaler.scale(torch.mean(torch.tensor(disc_losses, requires_grad=True))).backward()
                self.disc_scaler.step(self.discriminator_optimizer)
                losses['disc_loss'].append(disc_loss.item())
                disc_losses = []

            if self.verbose_every_n_steps and step % self.verbose_every_n_steps == 0:
                msg = TEXT_MSG_PER_EVERY_N_STEP.format(
                    epoch=self.current_epoch, step=step, mode='train',
                    gen_loss=np.average(losses['gen_loss'][-self.getting_average_by_last_n:]),
                    disc_loss=np.average(losses['disc_loss'][-self.getting_average_by_last_n:])
                )
                self.logger.info(msg)

            self._on_train_step_end()

        return np.average(losses['gen_loss']), np.average(losses['disc_loss'])

    def _on_train_step_end(self):
        self.gen_scaler.update()
        if self.discriminator:
            self.disc_scaler.update()

        self.datasets.update()

    def _on_val_step_end(self):
        self.datasets.update()

    @torch.no_grad()
    def _val(self, test_loader: bool = False) -> Tuple[float, float]:
        if test_loader:
            loader = self.datasets.test_loader
            mode = 'testing'
        else:
            mode = 'validation'
            loader = self.datasets.val_loader

        self.change_training_mode(train=False)
        losses = {
            'gen_loss': [],
            'disc_loss': []
        }
        metrics = defaultdict(list)

        for step, data in enumerate(tqdm(loader)):
            gen_loss, disc_loss, metrics_per_step = self._step(data, calculate_metrics=True)

            losses['gen_loss'].append(gen_loss.item())

            for metric, value in metrics_per_step.items():
                metrics[metric].append(value)

            if self.discriminator:
                assert disc_loss is not None, 'disc_loss is none'
                losses['disc_loss'].append(disc_loss.item())

            if self.verbose_every_n_steps and step % self.verbose_every_n_steps == 0:
                msg = TEXT_MSG_PER_EVERY_N_STEP.format(
                    epoch=self.current_epoch, step=step, mode=mode,
                    gen_loss=np.average(losses['gen_loss'][-self.getting_average_by_last_n:]),
                    disc_loss=np.average(losses['disc_loss'][-self.getting_average_by_last_n:])
                )
                self.logger.info(msg)

            self._on_val_step_end()

        final_metrics_msg = ''
        for metric, values in metrics.items():
            final_metrics_msg += f'{metric} = {np.mean(values)}\n'
        self.logger.info(final_metrics_msg)

        return np.average(losses['gen_loss']), np.average(losses['disc_loss'])

    def change_training_mode(self, train=True):
        if train:
            self.generator.train()
            if self.discriminator:
                self.discriminator.train()
        else:
            self.generator.eval()
            if self.discriminator:
                self.discriminator.eval()

    def _test(self):
        return self._val(test_loader=True)

    def on_epoch_end(self, losses: Dict[str, List[float]]):
        if self.generator_scheduler:
            self.generator_scheduler.step()

        if self.discriminator_scheduler:
            self.discriminator_scheduler.step()

        self.save_state()
        self.save_losses(losses)

    def save_losses(self, losses: Dict[str, List[float]]):
        df = pd.DataFrame.from_dict(losses)
        sns.set(style='darkgrid')
        plt.figure()
        sns.lineplot(x='epoch', y='train_gen_loss', data=df).figure.savefig(os.path.join(self.log_dir, "train_gen_loss.png"))
        if not df['train_disc_loss'].isna().any():
            plt.figure()
            sns.lineplot(x='epoch', y='train_disc_loss', data=df).figure.savefig(
                os.path.join(self.log_dir, "train_disc_loss.png"))
        plt.figure()
        sns.lineplot(x='epoch', y='val_gen_loss', data=df).figure.savefig(os.path.join(self.log_dir, "val_gen_loss.png"))
        if not df['val_disc_loss'].isna().any():
            plt.figure()
            sns.lineplot(x='epoch', y='val_disc_loss', data=df).figure.savefig(os.path.join(self.log_dir, "val_disc_loss.png"))
        plt.figure()
        df.to_csv(os.path.join(self.log_dir, 'losses_data.csv'))

    def save_info_about_training(self, text):
        if self.logger is not None:
            self.logger.info(text)
        else:
            with open(self.full_log_path, "a") as file:
                file.write(text + '\n')

    def save_state(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.log_dir, f"{self.current_epoch + 1}_epoch_checkpoint.pt")

        torch.save({
            'current_epoch': self.current_epoch + 1,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict() if self.discriminator else None,
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict() if self.discriminator else None,
            'discriminator_scheduler': self.discriminator_scheduler.state_dict() if self.discriminator_scheduler else None,
            'generator_scheduler': self.generator_scheduler.state_dict() if self.generator_scheduler else None,
            'gen_scaler': self.gen_scaler.state_dict(),
            'disc_scaler': self.disc_scaler.state_dict() if self.discriminator else None,
            'config': self.config,
        }, path)

    def load_state(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.log_dir, f"{self.current_epoch + 1}_epoch_checkpoint.pt")
        checkpoint = torch.load(path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
        self.gen_scaler.load_state_dict(checkpoint['gen_scaler'])

        if checkpoint['discriminator']:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.disc_scaler.load_state_dict(checkpoint['disc_scaler'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

        if checkpoint['discriminator_scheduler']:
            self.discriminator_scheduler.load_state_dict(checkpoint['discriminator_scheduler'])

        if checkpoint['generator_scheduler']:
            self.generator_scheduler.load_state_dict(checkpoint['generator_scheduler'])

        self.current_epoch = checkpoint['current_epoch']
        self.config = checkpoint['config']

    def visualize(
            self,
            pred_imgs: Tensor, real_imgs_LR: Tensor, real_imgs_SR: Tensor,
            generator_loss_on_step, discriminator_loss_on_step
    ):
        print_imgs = min(pred_imgs.shape[0], 4)

        for i in range(print_imgs):
            print('\npredicted')
            visualize_img_from_array(pred_imgs[i])
            print('real_img_LR')
            visualize_img_from_array(real_imgs_LR[i])
            print('real_img_SR')
            visualize_img_from_array(real_imgs_SR[i])

        print(f'generator_loss_on_step={generator_loss_on_step}')
        print(f'discriminator_loss_on_step={discriminator_loss_on_step}')
