import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

def get_gaussian_kernel(ksize, sigma):

    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d = gaussian(ksize, sigma)
    return window_1d

def get_gaussian_kernel2d(ksize, sigma):
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d

class GaussianBlur(nn.Module):

    def __init__(self, kernel_size, sigma):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = tuple([kernel_size] * 2)
        if isinstance(sigma, float):
            self.sigma = tuple([sigma] * 2)
        if len(self.kernel_size) is not 2 or len(self.sigma) is not 2:
            raise ValueError("Invalid kernel or sigma")
        self.padding = self.compute_zero_padding(self.kernel_size)
        self.kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma)

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        kernel = get_gaussian_kernel2d(kernel_size, sigma)
        return kernel

    @staticmethod
    def compute_zero_padding(kernel_size):
        return [(k - 1) // 2 for k in kernel_size]

    def forward(self, x):
        in_shape = x.shape
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(x)))
        if len(x.shape) == 3:
            # expand dim as batch_size=1
            x = x.view(1, *x.shape)
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        _, c, _, _ = x.shape
        tmp_kernel = self.kernel.to(x.device).to(x.dtype)
        kernel = tmp_kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, kernel, padding=self.padding, stride=1, groups=c).view(*in_shape)

def gaussian_blur(src, kernel_size=3, sigma=1.):
    return GaussianBlur(kernel_size, sigma)(src)
