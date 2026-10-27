from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock


class CifarResNet18(ResNet):
    def __init__(self, classes=10):
        super().__init__(BasicBlock,[2,2,2,2],num_classes=classes)
        self.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
        self.maxpool=nn.Identity()

    def forward_features(self,x,return_intermediate=False):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        layers={}
        for i,stage in enumerate([self.layer1,self.layer2,self.layer3,self.layer4],1):
            x=stage(x);layers[f'stage{i}']=x
        return layers if return_intermediate else self.avgpool(x).flatten(1)

    def forward(self,x):
        return self.fc(self.forward_features(x))
