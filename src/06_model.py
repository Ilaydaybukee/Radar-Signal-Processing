"""Küçük veri kümeleri için hafif, tek kanallı CNN."""
import torch.nn as nn
class SARClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.3),nn.Linear(64,num_classes))
    def forward(self,x): return self.classifier(self.features(x))
