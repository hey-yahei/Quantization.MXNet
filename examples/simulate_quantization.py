#-*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 hey-yahei
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from mxnet import cpu, gpu, nd
from mxnet.gluon import nn
from mxnet.gluon.data import Sampler, DataLoader
from mxnet.gluon.data.vision import CIFAR10
from mxnet.gluon.data.vision import transforms as T
from gluoncv.model_zoo import get_model, get_model_list
from gluoncv.data import ImageNet

import argparse
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("..")
from quantize import convert
from quantize.initialize import qparams_init
from quantize.distribution_calibrate import kl_calibrate, collect_feature_maps

__author__ = "YaHei"


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate for quantization.')
    # parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets',
    #                     help='training and validation pictures to use. (default: ~/.mxnet/datasets)')
    parser.add_argument('--model', type=str, default=None,
                        help='type of model to use. see vision_model for options. (required)')
    parser.add_argument('--print-model', action='store_true',
                        help='print the architecture of model.')
    parser.add_argument('--list-models', action='store_true',
                        help='list all models supported for --model.')
    parser.add_argument('--use-gpu', type=int, default=-1,
                        help='run model on gpu. (default: cpu)')
    parser.add_argument('--dataset', type=str, default="imagenet",
                        choices=['imagenet', 'cifar10'],
                        help='dataset to evaluate (default: imagenet)')
    parser.add_argument('--use-gn', action='store_true',
                        help='whether to use group norm.')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--use-se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--merge-bn', action='store_true',
                        help='merge batchnorm into convolution or not. (default: False)')
    parser.add_argument('--weight-bits-width', type=int, default=8,
                        help='bits width of weight to quantize into.')
    parser.add_argument('--input-signed', type=str, default="false",
                        help='quantize inputs into int(true) or uint(fasle). (default: false)')
    parser.add_argument('--input-bits-width', type=int, default=8,
                        help='bits width of input to quantize into.')
    parser.add_argument('--quant-type', type=str, default="layer",
                        choices=['layer', 'group', 'channel'],
                        help='quantize weights on layer/group/channel. (default: layer)')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='evaluate batch size per device (CPU/GPU). (default: 128)')
    parser.add_argument('--num-sample', type=int, default=5,
                        help='number of samples for every class in trainset. (default: 5)')
    parser.add_argument('--quantize-input-offline', action='store_true',
                        help='calibrate via EMA on trainset and quantize input offline.')
    parser.add_argument('--calib-mode', type=str, default="naive",
                        choices=['naive', 'kl'],
                        help='how to calibrate inputs. (default: naive)')
    parser.add_argument('--calib-epoch', type=int, default=3,
                        help='number of epoches to calibrate via EMA on trainset. (default: 3)')
    parser.add_argument('--disable-cudnn-autotune', action='store_true',
                        help='disable mxnet cudnn autotune to find the best convolution algorithm.')
    parser.add_argument('--eval-per-calib', action='store_true',
                        help='evaluate once after every calibration.')
    parser.add_argument('--exclude-first-conv', type=str, default="true",
                        choices=['false', 'true'],
                        help='exclude first convolution layer when quantize. (default: true)')
    parser.add_argument('--fixed-random-seed', type=int, default=7,
                        help='set random_seed for numpy to provide reproducibility. (default: 7)')
    opt = parser.parse_args()

    if opt.list_models:
        for key in get_model_list():
            print(key)
        exit(0)
    elif opt.model is None:
        print("error: --model is required")

    print()
    print('*'*25 + ' Settings ' + '*'*25)
    for k, v in opt.__dict__.items():
        print("{0: <25}: {1}".format(k, v))
    print('*'*(25*2+len(' Setting ')))
    print()
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

    # set random_seed for numpy
    np.random.seed(opt.fixed_random_seed)

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

    if opt.print_model:
        print('*'*25 + ' ' + opt.model + ' ' + '*'*25)
        print(net)
        print('*'*(25*2 + 2 + len(opt.model)))
        print()

    # convert model to quantization version
    convert_fn = {
        nn.Conv2D: convert.gen_conv2d_converter(
            quantize_input=True,
            fake_bn=opt.merge_bn,
            input_signed=opt.input_signed == 'true',
            weight_width=opt.weight_bits_width,
            input_width=opt.input_bits_width,
            quant_type=opt.quant_type
        ),
        nn.Dense: convert.gen_dense_converter(
            quantize_input=True,
            input_signed=opt.input_signed == 'true',
            weight_width=opt.weight_bits_width,
            input_width=opt.input_bits_width,
            quant_type=opt.quant_type
        ),
        # nn.Activation: convert.gen_act_converter(
        #     quantize_act=True,
        #     width=opt.input_bits_width
        # ),
        nn.Activation: None,
        nn.BatchNorm: convert.bypass_bn if opt.merge_bn else None
    }
    exclude_blocks = []
    if opt.exclude_first_conv == 'true':
        exclude_blocks.extend([net.features[0], net.features[1]])
    if model_name.startswith('mobilenetv2_'):
        exclude_blocks.append(net.output[0])
    if model_name.startswith('cifar_resnet'):
        exclude_blocks.extend([net.features[2][0].body[0], net.features[2][0].body[1]])
    print('*'*25 + ' Exclude blocks ' + '*'*25)
    for b in exclude_blocks:
        print(b.name)
    print('*'*(25*2 + len(' Exclude blocks ')))
    print()
    convert.convert_model(net, exclude=exclude_blocks, convert_fn=convert_fn, )

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
        if opt.dataset == 'imagenet':
            train_labels = [item[1] for item in train_dataset._data.items]
        elif opt.dataset == 'cifar10':
            train_labels = train_dataset._data._label
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            sampler=UniformSampler(classes, opt.num_sample, train_labels),
            num_workers=opt.num_workers,
            last_batch='keep'
        )

    # calibrate for input ranges and evaluate for simulation
    if opt.quantize_input_offline:
        if opt.calib_mode == "kl":
            print('*' * 25 + ' KL Calibration ' + '*' * 25)
            net.disable_quantize()
            input_levels = 2 ** ((opt.input_bits_width - 1) if opt.input_signed == "true" else opt.input_bits_width)
            min_bins, bins = input_levels, 2048
            # collect feature maps
            hist_collector, fm_max_collector = collect_feature_maps(net, bins=bins, loader=train_loader, ctx=ctx)
            # do calibration
            thresholds = {}
            quantized_blocks = net.collect_quantized_blocks()
            n_quantized_blocks = len(quantized_blocks)
            for i, m in enumerate(quantized_blocks):
                best_bins = kl_calibrate(hist_collector[m], levels=input_levels, min_bins=min_bins, bins=bins)
                thresholds[m] = (best_bins + 0.5) * (fm_max_collector[m] / bins)
                print(f"({i+1}/{n_quantized_blocks})\tBest threshold for {m.name}: {thresholds[m]}")
            # update input_max
            for m, th in thresholds.items():
                m.input_max.set_data(nd.array([th]))
            net.enable_quantize()
            print('*' * (25 * 2 + len(' KL Calibration ')))
            print()
        else:
            print('*' * 25 + ' Naive Calibration ' + '*' * 25)
            for i in range(opt.calib_epoch):
                net.quantize_input(enable=False)
                _ = evaluate(net, classes, train_loader, ctx=ctx, update_ema=True,
                             tqdm_desc="Calib[{}/{}]".format(i+1, opt.calib_epoch))
                if opt.eval_per_calib:
                    net.quantize_input(enable=True, online=False)
                    acc, avg_acc = evaluate(net, classes, eval_loader, ctx=ctx, update_ema=False,
                                            tqdm_desc="Eval[{}/{}]".format(i + 1, opt.calib_epoch))
                    print('{0: <8}: {1:2.2f}%'.format('acc', acc * 100))
                    print('{0: <8}: {1:2.2f}%'.format('avg_acc', avg_acc * 100))
                    print()
            print('*' * (25 * 2 + len(' Naive Calibration ')))
            print()
        if not opt.eval_per_calib:
            net.quantize_input(enable=True, online=False)
            acc, avg_acc = evaluate(net, classes, eval_loader, ctx=ctx, update_ema=False)
            print('*' * 25 + ' Result ' + '*' * 25)
            print('{0: <8}: {1:2.2f}%'.format('acc', acc * 100))
            print('{0: <8}: {1:2.2f}%'.format('avg_acc', avg_acc * 100))
            print('*' * (25 * 2 + len(' Result ')))
            print()
    else:
        net.quantize_input(enable=True, online=True)
        acc, avg_acc = evaluate(net, classes, eval_loader, ctx=ctx, update_ema=False)
        print('*'*25 + ' Result ' + '*'*25)
        print('{0: <8}: {1:2.2f}%'.format('acc', acc * 100))
        print('{0: <8}: {1:2.2f}%'.format('avg_acc', avg_acc * 100))
        print('*'*(25*2 + len(' Result ')))
        print()
