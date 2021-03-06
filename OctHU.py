import torch
import torch.nn as nn
import torch.nn.functional as F
from octconv import OctConv2d


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

def _cat(_inputs, dim=0, out=None):
  input1, input2 = _inputs
  hf1, lf1 = input1 if isinstance(input1, tuple) else (input1, None)
  hf2, lf2 = input2 if isinstance(input2, tuple) else (input2, None)

  hf = torch.cat((hf1, hf2), dim=dim) if hf1 is not None and hf2 is not None else hf1
  lf = torch.cat((lf1, lf2), dim=dim) if lf1 is not None and lf2 is not None else lf1
  if lf1 is None and lf2 is not None:
    lf = lf2
  if hf1 is None and hf2 is not None:
    hf = hf2
  
  if lf is not None:
    return (hf, lf)
  else:
    return hf

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

class OctHU(nn.Module):
  def __init__(self, num_classes=1000, init_weights=True, color=True, alpha=0):
    super(OctHU, self).__init__()
    self.planes = [16, 32, 64, 128, 256]
    self.alpha = alpha

    self.conv1 = self.make_layers(3 if color else 1, self.planes[0], first_layer=True)
    self.conv2 = self.make_layers(self.planes[0], self.planes[1])
    self.conv3 = self.make_layers(self.planes[1], self.planes[2])
    self.conv4 = self.make_layers(self.planes[2], self.planes[3])
    self.conv5 = self.make_layers(self.planes[3], self.planes[4])

    self.conv6 = self.make_layers(self.planes[4]+self.planes[3], self.planes[3])
    self.conv7 = self.make_layers(self.planes[3]+self.planes[2], self.planes[2])
    self.conv8 = self.make_layers(self.planes[2]+self.planes[1], self.planes[1])
    self.conv9 = self.make_layers(self.planes[1]+self.planes[0], self.planes[0])

    self.conv10 = OctConv2d(self.planes[0], 1, kernel_size=3, padding=1, alpha=(self.alpha,0))

    self.pool = _MaxPool2d(kernel_size=2, stride=2)

  def make_layers(self, inplanes, outplanes, first_layer=False):
    layers = []
    first_alpha = self.alpha if not first_layer else (0, self.alpha)
    layers.append(OctConv2d(inplanes, outplanes, kernel_size=3, padding=1, alpha=first_alpha))
    layers.append(_ReLU(inplace=True))
    layers.append(OctConv2d(outplanes, outplanes, kernel_size=3, padding=1, alpha=self.alpha))
    layers.append(_ReLU(inplace=True))

    return nn.Sequential(*layers)

  def forward(self, x):
    conv1 = self.conv1(x)
    pool1 = self.pool(conv1)

    conv2 = self.conv2(pool1)
    pool2 = self.pool(conv2)

    conv3 = self.conv3(pool2)
    pool3 = self.pool(conv3)

    conv4 = self.conv4(pool3)
    pool4 = self.pool(conv4)

    conv5 = self.conv5(pool4)

    up6 = F.interpolate(conv5, scale_factor=2)
    up6 = _cat((up6, conv4), dim=1)
    conv6 = self.conv6(up6)

    up7 = F.interpolate(conv6, scale_factor=2)
    up7 = _cat((up7, conv3), dim=1)
    conv7 = self.conv7(up7)

    up8 = F.interpolate(conv7, scale_factor=2)
    up8 = _cat((up8, conv2), dim=1)    
    conv8 = self.conv8(up8)

    up9 = F.interpolate(conv8, scale_factor=2)
    up9 = _cat((up9, conv1), dim=1)
    conv9 = self.conv9(up9)

    conv10 = self.conv10(conv9)

    return conv10