"""Regression tests for fresh warm-start dataset offsets."""

import torch

from dwm_pano.train import _InitialSamplerOffset, create_parser


class _EpochSampler(torch.utils.data.Sampler):
    def __init__(self, values):
        self.values = values
        self.epoch = None

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def set_epoch(self, epoch):
        self.epoch = epoch


def test_initial_sampler_offset_skips_only_the_first_pass():
    source = _EpochSampler(list(range(8)))
    sampler = _InitialSamplerOffset(source, 3)
    sampler.set_epoch(0)
    assert source.epoch == 0
    assert len(sampler) == 5
    assert list(sampler) == [3, 4, 5, 6, 7]
    assert len(sampler) == 8
    assert list(sampler) == list(range(8))


def test_parser_exposes_initial_data_offset():
    args = create_parser().parse_args([
        "-c", "config.json",
        "--initial-data-offset", "2500",
    ])
    assert args.initial_data_offset == 2500
    assert args.resume_data_offset == 0
