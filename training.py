import argparse

import torch

from data.dataset import DataSetMode, SRDatasets
from metrics.metrics import Metrics
from models.discriminator import Discriminator
from models.generator import Generator
from preprocessing.apply_albumentations import ApplyAlbumentation
import os

from utils import get_param_from_config, lock_deterministic, object_from_dict

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
config_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "configs"))

parser = argparse.ArgumentParser()
parser.add_argument('--config', action="store", dest="config",
                    default=os.path.join(config_dir, "train.yaml"),
                    type=str)

args = parser.parse_args()


def main(train_config: dict):
    lock_deterministic(train_config.seed)

    VALID_PATH = train_config.valid_path
    TRAIN_PATH = train_config.train_path
    TEST_PATH = train_config.test_path

    ROOT = train_config.root
    updates_per_epoch = train_config.updates_per_epoch

    scale_coef = train_config.scale_coef

    train_transforms = ApplyAlbumentation()
    valid_transforms = ApplyAlbumentation(prob_do_nothing=1.)

    train_dataloader_kwargs = train_config.train_dataloader
    valid_dataloader_kwargs = train_config.valid_dataloader
    train_dataloader_kwargs.update(shuffle=True)
    valid_dataloader_kwargs.update(shuffle=False)

    DEVICE = torch.device(train_config.device)

    ROOT_TO_SAVE_MODEL = train_config.root_to_save_model

    train_dataset = object_from_dict(train_config.dataset, csv_path=TRAIN_PATH, root_to_data=ROOT,
                                     scale_coef=scale_coef, dataloader_kwargs=train_dataloader_kwargs,
                                     augmentation=train_transforms, mode=DataSetMode.TRAIN,
                                     length=updates_per_epoch * train_config.train_dataloader.batch_size)

    valid_dataset = object_from_dict(train_config.dataset, csv_path=VALID_PATH, root_to_data=ROOT,
                                     scale_coef=scale_coef, dataloader_kwargs=valid_dataloader_kwargs,
                                     augmentation=valid_transforms, mode=DataSetMode.VALIDATION)

    test_dataset = object_from_dict(train_config.dataset, csv_path=TEST_PATH, root_to_data=ROOT,
                                    scale_coef=scale_coef, dataloader_kwargs=valid_dataloader_kwargs,
                                    augmentation=valid_transforms, mode=DataSetMode.TEST)

    sr_datasets = SRDatasets(
        train_dataset,
        valid_dataset,
        test_dataset,
    )

    if not os.path.exists(os.path.join(train_config.root_to_save_model)):
        os.mkdir(os.path.join(train_config.root_to_save_model))

    discriminator = None
    discriminator_optimizer = None
    discriminator_loss = None
    discriminator_scheduler = None
    if 'discriminator_model' in train_config:
        discriminator_model = object_from_dict(train_config.discriminator_model)
        discriminator = Discriminator(model=discriminator_model).to(DEVICE)
        discriminator_optimizer = object_from_dict(train_config.discriminator_optimizer, params=discriminator.parameters())
        discriminator_loss = object_from_dict(train_config.discriminator_loss).to(DEVICE)
        if 'discriminator_scheduler' in train_config:
            discriminator_scheduler = object_from_dict(
                train_config.discriminator_scheduler, optimizer=discriminator_optimizer
            )
    generator_model = object_from_dict(train_config.generator_model, n_super_resolution=scale_coef)
    generator_loss = object_from_dict(train_config.generator_loss).to(DEVICE)
    generator = Generator(model=generator_model).to(DEVICE)
    generator_optimizer = object_from_dict(
        train_config.generator_optimizer,
        params=generator.parameters()
    )
    generator_scheduler = None
    if 'generator_scheduler' in train_config:
        generator_scheduler = object_from_dict(train_config.generator_scheduler, optimizer=generator_optimizer)

    metrics = Metrics()

    trainer = object_from_dict(
        train_config.trainer,
        device=DEVICE,

        datasets=sr_datasets,
        log_dir=ROOT_TO_SAVE_MODEL,

        metrics=metrics,

        discriminator_loss=discriminator_loss,
        discriminator_optimizer=discriminator_optimizer,
        discriminator=discriminator,

        generator_loss=generator_loss,
        generator_optimizer=generator_optimizer,
        generator=generator,
        discriminator_scheduler=discriminator_scheduler,
        generator_scheduler=generator_scheduler,

        config=train_config,
    )

    trainer.train(train_config.n_epochs)


if __name__ == "__main__":
    train_cfg = get_param_from_config(args.config)
    main(train_cfg)
