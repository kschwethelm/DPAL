from torch import nn
from .nfnets import pretrained_nfnet

import os

class NFNetF0(nn.Module):
    def __init__(self, num_classes, weights_path="weights/F0_haiku.npz"):
        super(NFNetF0, self).__init__()
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, weights_path)
        self.model = pretrained_nfnet(path)
        self.model.linear = nn.Linear(self.model.linear.in_features, num_classes)
        self.model.num_classes = num_classes

    def forward(self, x):
        return self.model(x)
    
    def exclude_from_weight_decay(self, name:str) -> bool:
        return self.model.exclude_from_weight_decay(name)

    def exclude_from_clipping(self, name: str) -> bool:
        return self.model.exclude_from_clipping(name)