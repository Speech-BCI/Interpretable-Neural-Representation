import torch
from inspect import isfunction
import math
import warnings



def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def calculate_num_patches(spec_size, patch_size, overlap_ratio):
    stride_size = divide_tuple(patch_size, (1 - overlap_ratio))
    height_out = (spec_size[0] - patch_size[0]) // stride_size[0] + 1
    width_out= (spec_size[1] - patch_size[1]) // stride_size[1] + 1
    num_patches = height_out * width_out

    total_padding_height = (stride_size[0] * (height_out - 1)) + patch_size[0] - spec_size[0]
    total_padding_width = (stride_size[1] * (width_out - 1)) + patch_size[1] - spec_size[1]

    padding_height = max(0, total_padding_height) // 2
    padding_width = max(0, total_padding_width) // 2

    return num_patches, stride_size, height_out, width_out

def adaptive_calculate_num_patches(spec_size, patch_size, overlap_ratio):
    stride_size_default = divide_tuple(patch_size, (1 - overlap_ratio))
    stride_size = (patch_size[0], stride_size_default[1])
    height_out = (spec_size[0] - patch_size[0]) // stride_size[0] + 1
    width_out= (spec_size[1] - patch_size[1]) // stride_size[1] + 1
    num_patches = height_out * width_out
    return num_patches, stride_size, height_out, width_out


def divide_tuple(tup, factor):
    return tuple(int(element * factor) for element in tup)





## https://github.com/YuanGongND/ssast
def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    # Examples:
    #     >>> w = torch.empty(3, 5)
    #     >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


