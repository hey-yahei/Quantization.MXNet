#-*- coding: utf-8 -*-

from mxnet import cpu, gpu, nd
from mxnet.gluon import nn
from mxnet.gluon.data import Sampler, DataLoader
from mxnet.gluon.data.vision import CIFAR10
from mxnet.gluon.data.vision import transforms as T
from gluoncv.model_zoo import get_model
from gluoncv.data import ImageNet

import argparse
import numpy as np
from tqdm import tqdm
import warnings

import sys
sys.path.append("..")
from quantize import convert
from quantize.initialize import qparams_init

__author__ = "YaHei"

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate for quantization.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='training and validation pictures to use.')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--use-gpu', type=int, default=-1,
                        help='run model on gpu.')
    parser.add_argument('--dataset', type=str, default="imagenet",
                        choices=['imagenet', 'cifar10'],
                        help='dataset to evaluate')
    parser.add_argument('--use-gn', action='store_true',
                        help='whether to use group norm.')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--use-se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--fake-bn', action='store_true',
                        help='use fake batchnorm or not.')
    parser.add_argument('--weight-dtype', type=str, default="uint",
                        choices=['int', 'uint'],
                        help='data type to simulate for weights')
    parser.add_argument('--weight-bits-width', type=int, default=8,
                        help='bits width of weight to quantize into.')
    parser.add_argument('--weight-one-side', type=str, default="false",
                        choices=['false', 'true'],
                        help='quantize weights as one-side uint or not.')
    # parser.add_argument('--weight-quantize-via-train', action='store_true',
    #                     help='regard weight ranges as parameters to train.')
    parser.add_argument('--input-dtype', type=str, default="uint",
                        choices=['float', 'int', 'uint'],
                        help='data type to simulate for inputs.')
    parser.add_argument('--input-bits-width', type=int, default=8,
                        help='bits width of input to quantize into.')
    parser.add_argument('--input-one-side', type=str, default="true",
                        choices=['false', 'true'],
                        help='quantize inputs as one-side uint or not.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='evaluate batch size per device (CPU/GPU).')
    parser.add_argument('--num-sample', type=int, default=10,
                        help='number of samples for every class in trainset.')
    parser.add_argument('--quantize-input-offline', action='store_true',
                        help='calibrate via EMA on trainset and quantize input offline.')
    parser.add_argument('--calib-epoch', type=int, default=3,
                        help='number of epoches to calibrate via EMA on trainset.')
    parser.add_argument('--show-warning', action='store_true',
                        help='show warning.')
    parser.add_argument('--disable-cudnn-autotune', action='store_true',
                        help='disable mxnet cudnn autotune to find the best convolution algorithm.')
    parser.add_argument('--eval-per-calib', action='store_true',
                        help='evaluate once after every clibration.')


    opt = parser.parse_args()
    return opt

def evaluate(net, num_class, dataloader, ctx, update_ema=False, tqdm_desc="Eval"):
    correct_counter = nd.zeros(num_class)
    label_counter = nd.zeros(num_class)
    test_num_correct = 0

    with tqdm(total=len(dataloader), desc=tqdm_desc) as pbar:
        for i, (X, y) in enumerate(dataloader):
            X = X.as_in_context(ctx)
            y = y.as_in_context(ctx)
            outputs = net(X)
            if update_ema:
                net.update_ema()
            # collect predictions
            pred = outputs.argmax(axis=1)
            test_num_correct += (pred == y.astype('float32')).sum().asscalar()
            pred = pred.as_in_context(cpu())
            y = y.astype('float32').as_in_context(cpu())
            for p, gt in zip(pred, y):
                label_counter[gt] += 1
                if p == gt:
                    correct_counter[gt] += 1
            # update tqdm
            pbar.update(1)
    # calculate acc and avg_acc
    eval_acc = test_num_correct / label_counter.sum().asscalar()
    eval_acc_avg = (correct_counter / (label_counter + 1e-10)).mean().asscalar()
    return eval_acc, eval_acc_avg

class UniformSampler(Sampler):
    def __init__(self, classes, num_per_class, labels):
        self._classes = classes
        self._num_per_class = num_per_class
        self._labels = labels
    def __iter__(self):
        sample_indices = []
        label_counter = np.zeros(self._classes)
        shuffle_indices = np.arange(len(self._labels))
        np.random.shuffle(shuffle_indices)
        for idx in shuffle_indices:
            label = self._labels[idx]
            if label_counter[label] < self._num_per_class:
                sample_indices.append(idx)
                label_counter[label] += 1
            if label_counter.sum() == self._classes * self._num_per_class:
                break
        for idx, cnt in enumerate(label_counter):
            if cnt < self._num_per_class:
                raise ValueError("Number of samples for class {} is {} < {}".format(idx, cnt, self._num_per_class))
        return iter(sample_indices)
    def __len__(self):
        return self._classes * self._num_per_class

if __name__ == "__main__":
    opt = parse_args()

    # show warning
    if not opt.show_warning:
        warnings.filterwarnings("ignore")
    # disable autotune
    if opt.disable_cudnn_autotune:
        import os
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    # get model
    model_name = opt.model
    classes = 10 if opt.dataset == 'cifar10' else 1000
    kwargs = {
        'pretrained': True,
        'classes': classes
    }
    if opt.use_gn:
        from gluoncv.nn import GroupNorm
        kwargs['norm_layer'] = GroupNorm
    if model_name.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm
    elif model_name.startswith('resnext'):
        kwargs['use_se'] = opt.use_se
    if opt.last_gamma:
        kwargs['last_gamma'] = True
    net = get_model(model_name, **kwargs)

    # convert model to quantization version
    conv_kwargs = {
        'quantize_input': opt.input_dtype != 'float',
        'fake_bn': opt.fake_bn,
        'weight_signed': opt.weight_dtype == 'int',
        'weight_width': opt.weight_bits_width,
        'weight_one_side': opt.weight_one_side == 'true',
        'weight_training': False,
        'input_signed': opt.input_dtype == 'int',
        'input_width': opt.input_bits_width,
        'input_one_side': opt.input_one_side == 'true'
    }
    convert_fn = {
        nn.Conv2D: convert.gen_conv2d_converter(**conv_kwargs),
        nn.BatchNorm: convert.bypass_bn if opt.fake_bn else None
    }
    convert.convert_model(net, convert_fn=convert_fn)

    # initialize for quantization parameters and reset context
    qparams_init(net)
    ctx = gpu(opt.use_gpu) if opt.use_gpu != -1 else cpu()
    net.collect_params().reset_ctx(ctx)

    # construct transformer
    if opt.dataset == 'imagenet':
        eval_transformer = T.Compose([
            T.Resize(256, keep_ratio=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        eval_transformer = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    # fetch dataset and dataloader
    dataset = ImageNet if opt.dataset == 'imagenet' else CIFAR10
    eval_dataset = dataset(train=False).transform_first(eval_transformer)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        last_batch='keep'
    )
    if opt.quantize_input_offline:
        train_dataset = dataset(train=True).transform_first(eval_transformer)
        train_labels = [item[1] for item in train_dataset._data.items]
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            sampler=UniformSampler(classes, opt.num_sample, train_labels),
            num_workers=opt.num_workers,
            last_batch='keep'
        )

    # calibrate for input ranges and evaluate for simulation
    if opt.quantize_input_offline:
        for i in range(opt.calib_epoch):
            _ = evaluate(net, classes, train_loader, ctx=ctx, update_ema=True,
                         tqdm_desc="Calib[{}/{}]".format(i+1, opt.calib_epoch))
            if opt.eval_per_calib:
                net.quantize_input_offline()
                print(evaluate(net, classes, eval_loader, ctx=ctx, update_ema=False,
                               tqdm_desc="Eval[{}/{}]".format(i + 1, opt.calib_epoch)))
                net.quantize_input_online()
    if not opt.eval_per_calib:
        net.quantize_input_offline()
        print(evaluate(net, classes, eval_loader, ctx=ctx, update_ema=False))
