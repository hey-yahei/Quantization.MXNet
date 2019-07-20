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

import copy
import numpy as np

from mxnet import nd

__all__ = ['kl_calibrate']
__author__ = 'YaHei'


def _discrete_histogram(feature_maps, bins, max_=None):
    # Check min and max
    assert np.min(feature_maps) >= 0., "Activation should >=0"
    if max_ is None:
        max_ = np.max(feature_maps)
    assert max_ > 0, "Bad distribution: all zero-value"

    # Clip and quantize
    feature_maps = feature_maps.reshape(-1).clip(0, max_)
    feature_maps = feature_maps[feature_maps != 0]   # ignore zero-value
    scales = bins / (max_ + 1e-5)   # +1e-5 to make sure that scales<1 and quantized_data<levels
    quantized_data = (feature_maps * scales).astype("int32")

    # Call np.bincount to get histogram
    hist = np.bincount(quantized_data, minlength=bins)

    return hist.astype("float32"), max_


def _collect_feature_maps(net, bins, loader, ctx):
    quantized_blocks = net.collect_quantized_blocks()

    """ Add hooks to quantized blocks """
    hooks = []
    fm_collector = {}
    for blk in quantized_blocks:
        def _collect(m, x, y):
            nonlocal fm_collector
            x = x[0]
            m_collector = fm_collector.setdefault(m, [])
            m_collector.append(x.asnumpy())
        h = blk.register_forward_hook(_collect)
        hooks.append(h)

    """ Collect feature maps """
    fm_max_collector = {}
    hist_collector = {}
    i = 0
    for X, _ in loader:
        i += 1
        print(i)
        X = X.as_in_context(ctx)
        _ = net(X)
        # Deal with feature maps data
        for m, fm in fm_collector.items():
            fm = np.concatenate(fm, axis=0)
            fm_max = fm_max_collector.get(m, None)
            hist, max_ = _discrete_histogram(fm, bins, fm_max)
            # First chunk, set the max activation, times max_factor to make it redundant.
            if fm_max is None:
                fm_max_collector[m] = max_
            # Update collectors
            last_hist = hist_collector.get(m, 0)
            hist_collector[m] = last_hist + hist

        # reset collector
        fm_collector.clear()

    """ Delete hooks """
    for h in hooks:
        h.detach()

    return hist_collector, fm_max_collector


def _kl_calibrate_once(data, levels, min_bins, bins):
    # Check
    assert min_bins >= levels, f"min_bins should be greater than levels ({min_bins} vs. {levels})"

    # Search the best bins with min_divergence
    min_divergence = np.inf
    best_bins = min_bins
    for i in range(min_bins, bins):
        # Get reference distribution
        # ... P = [bin[0], bin[1], ..., bin[i-1]]
        # ... bin[i-1] += sum([bin[i], ..., bin[bins])
        # ... then normalize P
        ref_distribution = copy.deepcopy(data[:i])
        ref_distribution[i-1] += sum(data[i:])
        # ref_distribution /= sum(ref_distribution)

        # Get candidate distribution
        # ... Q = quantize [bin[0], ..., bin[i-1]] to several levels
        bin_idx = (levels * np.arange(i) / i).astype("int32")
        cand_distribution = np.zeros(shape=levels)
        for j, idx in enumerate(bin_idx):
            cand_distribution[idx] += data[j]
        # ... expand Q to i bins
        cand_distribution_expand = np.zeros(shape=i)
        bin_count = np.bincount(bin_idx, minlength=levels)
        start_idx = 0
        for j, count in enumerate(bin_count):
            end_idx = start_idx + count
            nonzero_mask = (data[start_idx:end_idx] != 0)
            n_nonzero = sum(nonzero_mask)
            # if P(x) == 0, also set Q(x) = 0
            if n_nonzero == 0:
                cand_distribution_expand[start_idx:end_idx] = 0
            else:
                cand_distribution_expand[start_idx:end_idx] = cand_distribution[j] / n_nonzero
                cand_distribution_expand[start_idx:end_idx] *= nonzero_mask
            start_idx = end_idx
        # ... then normalize Q_expand
        print(i, sum(ref_distribution), sum(cand_distribution_expand))
        exit()
        # ref_distribution /= sum(ref_distribution)
        cand_distribution_expand /= sum(cand_distribution_expand)

        # Compute KL-divergence and search the least one (ignore zero-probability)
        ref_distribution = ref_distribution[cand_distribution_expand != 0]
        cand_distribution_expand = cand_distribution_expand[cand_distribution_expand != 0]
        divergence = sum(ref_distribution * np.log(ref_distribution / cand_distribution_expand))
        if divergence < min_divergence:
            min_divergence = divergence
            best_bins = i

    return best_bins


def _print(log_str, enable):
    if enable:
        print(log_str)


def kl_calibrate(net, calib_loader, ctx, levels=128, min_bins=None, bins=2048, enable_log=True):
    if min_bins is None:
        min_bins = levels

    _print("Collecting feature maps ...", enable_log)
    hist_collector, fm_max_collector = _collect_feature_maps(net, bins, calib_loader, ctx)

    thresholds = {}
    quantized_blocks = net.collect_quantized_blocks()
    n_quantized_blocks = len(quantized_blocks)
    for i, m in enumerate(quantized_blocks):
        best_bins = _kl_calibrate_once(hist_collector[m], levels, min_bins, bins)
        thresholds[m] = (best_bins + 0.5) * (fm_max_collector[m] / bins)
        _print(f"({i+1}/{n_quantized_blocks})Best threshold for {m.name}: {thresholds[m]}", enable_log)
    _print("", enable_log)

    return thresholds
