import torch.nn as nn

class Module(nn.Module):
  def __init__(self):
    super().__init__()

  def initialize_weights(self, mode='fan_in'):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
      elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
