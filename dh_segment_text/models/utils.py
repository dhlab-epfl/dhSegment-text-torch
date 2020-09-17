import torch


def conv2d_extra_params(conv2d: torch.nn.Conv2d):
    return {
        "out_channels": conv2d.out_channels,
        "kernel_size": conv2d.kernel_size,
        "stride": conv2d.stride,
        "padding": conv2d.padding,
        "dilation": conv2d.dilation,
        "groups": conv2d.groups,
        "bias": False if conv2d.bias is None else True,
        "padding_mode": conv2d.padding_mode,
    }