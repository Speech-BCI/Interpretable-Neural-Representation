import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_kernel(kernel_height, kernel_width, std_dev):
    """
    Create a 2D Gaussian kernel with different height and width.
    :param kernel_height: Height of the kernel
    :param kernel_width: Width of the kernel
    :param std_dev: Standard deviation of the Gaussian
    :return: 2D Gaussian kernel as a tensor
    """
    # Create 1D Gaussian distribution for height and width
    t_height = torch.linspace(-(kernel_height - 1) / 2., (kernel_height - 1) / 2., kernel_height)
    t_width = torch.linspace(-(kernel_width - 1) / 2., (kernel_width - 1) / 2., kernel_width)

    gauss_height = torch.exp(-0.5 * (t_height / std_dev) ** 2)
    gauss_width = torch.exp(-0.5 * (t_width / std_dev) ** 2)

    gauss_height = gauss_height / gauss_height.sum()
    gauss_width = gauss_width / gauss_width.sum()

    # Create 2D Gaussian kernel
    gaussian_kernel = gauss_height.unsqueeze(1) * gauss_width.unsqueeze(0)
    return gaussian_kernel / gaussian_kernel.sum()

def gaussian_smoothing_2d(data, kernel_height, kernel_width, std_dev):
    """
    Apply 2D Gaussian smoothing on 4D data (batch, channel, freq, time).
    :param data: Input data as a 4D tensor
    :param kernel_size: Size of the Gaussian kernel
    :param std_dev: Standard deviation of the Gaussian
    :return: Smoothed data
    """
    # Create Gaussian kernel
    gaussian_kernel = create_gaussian_kernel(kernel_height, kernel_width, std_dev).to(data.device)
    gaussian_kernel = gaussian_kernel.expand(data.shape[1], 1, kernel_height, kernel_width)
    # gaussian_kernel = gaussian_kernel.expand(data.shape[1], 1, kernel_size, kernel_size)

    # Padding for same output size
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # Apply convolution
    smoothed_data = torch.nn.functional.conv2d(data, gaussian_kernel, padding=(padding_height, padding_width), groups=data.shape[1])

    return smoothed_data


def get_conv(
        in_dim,
        out_dim,
        kernel_size,
        stride,
        padding,
        zero_bias=True,
        zero_weights=False,
        groups=1,
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=1)

def get_5x5(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 5, 1, 2, zero_bias, zero_weights, groups=1)

def get_1x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, (1, 3), 1, 1, zero_bias, zero_weights, groups=1)
def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=1)



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()




class Res_Block(nn.Module):
    def __init__(
            self,
            in_width,
            middle_width,
            out_width,
            down_rate=None,
            residual=False,
            zero_last=False,
            down_channel=False,
            first_block=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.down_channel = down_channel
        self.first_block = first_block
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)
        self.gn =  nn.GroupNorm(num_groups=4, num_channels=middle_width)




    def forward(self, x):
        # print("in ResBlock x shape: ", x.shape)
        if self.down_channel is False:
            if self.first_block:
                xhat = self.c1(x)
            else:
                xhat = self.c1(F.gelu(x))
            xhat = self.gn(xhat)
            xhat = self.c2(F.gelu(xhat))
            xhat = self.gn(xhat)
            xhat = self.c3(F.gelu(xhat))
            xhat = self.gn(xhat)
            xhat = self.c4(F.gelu(xhat))
            # print("in ResBlock xhat shape: ", xhat.shape)

            out = x + xhat if self.residual else xhat

            if self.down_rate is not None:
                # out = F.avg_pool2d(out, kernel_size=(self.down_rate // 2, self.down_rate), stride=(self.down_rate // 2, self.down_rate))
                out = F.avg_pool2d(out, kernel_size=(1, self.down_rate), stride=(1, self.down_rate))
            return out
        else:
            xhat = self.c1(F.gelu(x))
            xhat = self.gn(xhat)
            xhat = self.c2(F.gelu(xhat))
            xhat = self.gn(xhat)
            xhat = self.c3(F.gelu(xhat))
#             xhat = self.gn(xhat)
            out = self.c4(x) + xhat
            if self.down_rate is not None:
                out = F.avg_pool2d(out, kernel_size=(1, self.down_rate), stride=(1, self.down_rate))
            return out

def build_res_blocks(input_channel, num_blocks, down_rate=2):
    blocks = []
    for i in range(num_blocks):
        down_rate_block = down_rate if i == num_blocks - 1 else None
        blocks.append(
            Res_Block(
                input_channel,
                round(0.5 * input_channel),
                input_channel,
                down_rate=down_rate_block,
                residual=True,
                first_block=(i == 0)
            )
        )
    return blocks

class CNN_feature_encoder(nn.Module):
    def __init__(self, input_channel, down_rate = 2, repreated_blocks=3, out_channel=32):
        super().__init__()

        out_channel = [input_channel, out_channel*2, out_channel]

        blocks = []

        blocks.extend(build_res_blocks(out_channel[0], repreated_blocks, down_rate))
        blocks.append(nn.Conv2d(out_channel[0], out_channel[1], 1, stride=1, padding=0, bias=False))


        blocks.extend(build_res_blocks(out_channel[1], repreated_blocks, down_rate))
        blocks.append(nn.Conv2d(out_channel[1], out_channel[2], 1, stride=1, padding=0, bias=False))

        self.block_mod = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.block_mod(input)
        return x
