import torch.nn as nn
import torch.nn.functional as F
from torch import mul
import math
import torch

__all__ = ['OctConv2d', '_MaxPool2d', '_BatchNorm2d', '_ReLU']

class OctConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 alpha=0.5,
                 bias=False,
                 dilation=1,
                 groups=1,
                 ):

        super(OctConv2d, self).__init__()

        assert isinstance(in_channels, int) and in_channels > 0
        assert isinstance(out_channels, int) and out_channels > 0
        assert isinstance(kernel_size, int) and kernel_size > 0
        assert stride in {1, 2}, "Only strides of 1 and 2 are currently supported"

        if isinstance(alpha, tuple):
            assert len(alpha) == 2
            assert all([0 <= a <= 1 for a in alpha]), "Alphas must be in interval [0, 1]"
            self.alpha_in, self.alpha_out = alpha
        else:
            assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
            self.alpha_in = alpha
            self.alpha_out = alpha
        
        conv_type = nn.Conv2d
        
        # in_channels
        in_ch_hf = int((1 - self.alpha_in) * in_channels)
        self.in_channels = {
            'high': in_ch_hf,
            'low': in_channels - in_ch_hf
        }

        # out_channels
        out_ch_hf = int((1 - self.alpha_out) * out_channels)
        self.out_channels = {
            'high': out_ch_hf,
            'low': out_channels - out_ch_hf
        }

        # groups
        if in_channels == groups:
          groups_hf = int((1 - self.alpha_in) * groups)
          self.groups = {
              'high': groups_hf,
              'low': groups - groups_hf
          }
        else:
          self.groups = {
              'high': groups,
              'low': groups
          }

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.conv_h2h = conv_type(in_channels=self.in_channels['high'],
                                  out_channels=self.out_channels['high'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=bias,
                                  dilation=dilation,
                                  groups=self.groups['high']) \
            if not (self.alpha_in == 1 or self.alpha_out == 1) else None

        self.conv_h2l = conv_type(in_channels=self.in_channels['high'],
                                  out_channels=self.out_channels['low'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=bias,
                                  dilation=dilation,
                                  groups=self.groups['high']) \
            if not (self.alpha_in == 1 or self.alpha_out == 0) else None

        self.conv_l2h = conv_type(in_channels=self.in_channels['low'],
                                  out_channels=self.out_channels['high'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=bias,
                                  dilation=dilation,
                                  groups=self.groups['low']) \
            if not (self.alpha_in == 0 or self.alpha_out == 1) else None

        self.conv_l2l = conv_type(in_channels=self.in_channels['low'],
                                  out_channels=self.out_channels['low'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=bias,
                                  dilation=dilation,
                                  groups=self.groups['low']) \
            if not (self.alpha_in == 0 or self.alpha_out == 0) else None

    def forward(self, x):
        x_h, x_l = x if isinstance(x, tuple) else (x, None)

        self._check_inputs(x_h, x_l)
        x_h2h, x_h2l = None, None
        x_l2l, x_l2h = None, None
        
        

        # High -> High
        if x_h is not None:
            x_h = self.pool(x_h) if (self.out_channels['high'] > 0 and self.stride == 2) else x_h
            x_h2h = self.conv_h2h(x_h) if self.out_channels['high'] > 0 else None

            # High -> Low
            x_h2l = self.pool(x_h) if (self.out_channels['low'] > 0 and x_h is not None) else x_h
            x_h2l = self.conv_h2l(x_h2l) if (self.out_channels['low'] > 0 and x_h is not None) else None

        if x_l is not None:
            # Low -> Low
            x_l2l = self.pool(x_l) if (self.out_channels['low'] > 0 and self.stride == 2) else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.out_channels['low'] > 0 else None

            # Low -> High
            x_l2h = self.conv_l2h(x_l) if self.out_channels['high'] > 0 else None
            shape_x_l2h = x_h2h.shape[-2:] if x_h2h is not None else [i*2 for i in x_l.shape[-2:]]
            x_l2h = F.interpolate(x_l2h, size=shape_x_l2h) \
                if (self.out_channels['high'] > 0 and self.stride == 1) else x_l2h
        if x_l2h is None and x_h2h is not None:
            x_h = x_h2h
        else:
            x_l2h = F.interpolate(x_l2h, size=x_h2h.shape[-2:]) if (x_h2h is not None and x_h2h.shape != x_l2h.shape) else x_l2h            
            x_h = x_h2h + x_l2h if x_h2h is not None else x_l2h
        if x_h2l is None and x_l2l is not None:
            x_l = x_l2l
        else:
            x_h2l = F.interpolate(x_h2l, size=x_l2l.shape[-2:]) if (x_l2l is not None and x_l2l.shape != x_h2l.shape) else x_h2l            
            x_l = x_l2l + x_h2l if x_l2l is not None else x_h2l

        output = (x_h, x_l)

        return output[0] if output[1] is None else output

    def _check_inputs(self, x_h, x_l):
        if x_h is not None:
            assert x_h.dim() == 4

        if x_l is not None:
            assert x_l.dim() == 4

        #print("octonv.py 160 | in_channels['high'] & x_h.shape[1]",self.in_channels['high'], x_h.shape[1], self.in_channels['high'] == x_h.shape[1])
        #print("alphas: ", self.alpha_in, self.alpha_out)
        if self.in_channels['high'] > 0:
            assert x_h.shape[1] == self.in_channels['high']

        if self.in_channels['low'] > 0:
            assert x_l.shape[1] == self.in_channels['low']

    def __repr__(self):
        s = """{}(in_channels=(low: {}, high: {}), out_channels=(low: {}, high: {}),
          kernel_size=({kernel}, {kernel}), stride=({stride}, {stride}),
          padding={}, alphas=({}, {}), bias={})""".format(
            self._get_name(), self.in_channels['low'], self.in_channels['high'],
            self.out_channels['low'], self.out_channels['high'],
            self.padding, self.alpha_in, self.alpha_out, self.bias,
            kernel=self.kernel_size, stride=self.stride)

        return s


class _MaxPool2d(nn.Module):
  def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    super(_MaxPool2d, self).__init__()
    self.maxpool = nn.MaxPool2d(kernel_size, 
                    stride=stride, 
                    padding=padding, 
                    dilation=dilation, 
                    return_indices=return_indices, ceil_mode=ceil_mode)
  def forward(self, x):
    if isinstance(x, tuple):
        hf, lf = x
        hf = self.maxpool(hf) if type(hf) == torch.Tensor else hf
        lf = self.maxpool(lf) if type(lf) == torch.Tensor else lf
        return hf, lf
    else:
        return self.maxpool(x)


class _BatchNorm2d(nn.Module):
  def __init__(self, num_features, alpha_in=0, alpha_out=0, eps=1e-5, momentum=0.1, affine=True,
               track_running_stats=True):
    super(_BatchNorm2d, self).__init__()
    hf_ch = int(num_features * (1 - alpha_out))
    lf_ch = num_features - hf_ch
    self.bnh = nn.BatchNorm2d(hf_ch)
    self.bnl = nn.BatchNorm2d(lf_ch)
  def forward(self, x):
    if isinstance(x, tuple):
        hf, lf = x
        hf = self.bnh(hf) if type(hf) == torch.Tensor else hf
        lf = self.bnl(lf) if type(lf) == torch.Tensor else lf
        return hf, lf
    else:
        return self.bnh(x)


class _ReLU(nn.Module):
  def __init__(self, inplace=False):
    super(_ReLU, self).__init__()
    self.relu = nn.ReLU(inplace=inplace)
  def forward(self, x):
    if isinstance(x, tuple):
        hf, lf = x
        hf = self.relu(hf) if type(hf) == torch.Tensor else hf
        lf = self.relu(lf) if type(lf) == torch.Tensor else lf
        return hf, lf
    else:
        return self.relu(x)
