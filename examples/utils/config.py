#-*- coding: utf-8 -*-

import logging
import os

__all__ = ['ModelConfig', "TrainConfig", "QTrainConfig"]


class Config(object):
    def __init__(self, logger=None, *args, **kwargs):
        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._kwargs = kwargs
        self._apply_args()
        self._print_info()

    def _apply_args(self):
        for k, v in self._kwargs.items():
            if not hasattr(self, k):
                raise ValueError("Get unknown parameter: " + k)
            else:
                setattr(self, k, v)

    def _print_info(self):
        # Print custom parameters first
        default = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                if k in self._kwargs:
                    self._logger.info("{} => {}".format(k, v))
                else:
                    default.append((k, v))
        for p in default:
            self._logger.info("{} => {} (default)".format(*p))


class ModelConfig(Config):
    def __init__(self, logger=None, *args, **kwargs):
        # output shape
        self.num_class = 1000
        super(ModelConfig, self).__init__(logger, *args, **kwargs)


class TrainConfig(Config):
    def __init__(self, logger=None, *args, **kwargs):
        # Train
        self.max_steps = 300000
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.train_num_prefetch_workers = 4
        self.val_num_prefetch_workers = 4
        # Record
        self.checkpoint_dir = "./checkpoints"
        self.tb_main_tag = "train_config"
        self.checkpoint_prefix = self.tb_main_tag
        self.train_record_per_steps = 100
        self.val_per_steps = 200
        self.spotter_starts_at = 2000
        self.spotter_window_size = 10
        self.patience = 30
        self.snapshot_per_steps = 2000
        super(TrainConfig, self).__init__(logger, *args, **kwargs)
        # create directory if need
        if not os.path.exists(self.checkpoint_dir):
            logger.info("Create a new directory", self.checkpoint_dir)
            os.mkdir(self.checkpoint_dir)

    def summary(self, trainset, valset):
        logger = logging.getLogger(self.__class__.__name__ + ".summary")
        # summary
        train_size = len(trainset)
        val_size = len(valset)
        logger.info("trainset size => {}".format(train_size))
        logger.info("valset size => {}".format(val_size))
        steps_per_epoch = train_size / self.train_batch_size
        logger.info("{} steps for per epoch (BATCH_SIZE={})".format(steps_per_epoch, self.train_batch_size))
        logger.info("record per {} steps ({} samples, {} times per epoch)".format(
                                                                self.train_record_per_steps,
                                                                self.train_record_per_steps * self.train_batch_size,
                                                                steps_per_epoch / self.train_record_per_steps))
        logger.info("evaluate per {} steps ({} times per epoch)".format(
                                                                self.val_per_steps,
                                                                steps_per_epoch / self.val_per_steps))
        logger.info("spotter start at {} steps ({} epoches)".format(
                                                                self.spotter_starts_at,
                                                                self.spotter_starts_at / steps_per_epoch))
        logger.info("size of spotter window is {} ({} steps)".format(
                                                                self.spotter_window_size,
                                                                self.spotter_window_size * self.val_per_steps))
        logger.info("max patience: {} ({} steps; {} samples; {} epoches)".format(self.patience,
                                                            self.patience * self.val_per_steps,
                                                            self.patience * self.val_per_steps * self.train_batch_size,
                                                            self.patience * self.val_per_steps / steps_per_epoch))
        logger.info("snapshot per {} steps ({} times per epoch)".format(
                                                                self.snapshot_per_steps,
                                                                steps_per_epoch / self.snapshot_per_steps))


class QTrainConfig(TrainConfig):
    def __init__(self, logger=None, *args, **kwargs):
        self.param_file = "param_file"
        self.quant_offline_after = 10000
        super(QTrainConfig, self).__init__(logger, *args, **kwargs)

