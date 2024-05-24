# model.resnet.py
### ライブラリ
import torch
import torch.nn as nn

from torchvision import models


### クラス定義
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # 新規作成
        # PSPNetで使用するResNetの最初の畳み込み層の定義を行う
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, stride=2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使いまわし
        # PyTorchで実装済みのResNetを読み込み、必要なところだけを使用する
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

        # 変更
        # Layer1をPSPNetで使用するResNetの実装に合わせるため、一部、Conv2dのパラメータを変更する
        self.layer1[0].conv1 = nn.Conv2d(in_channels=128, out_channels=64, stride=1, bias=False, kernel_size=1)
        self.layer1[0].downsample[0] = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


### 関数定義
def resnet50(pretrained=False):
    model = ResNet()
    
    if pretrained is True:
        model_path = '../initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        
    return model


# 動作確認
if __name__ == '__main__':
    input = torch.rand(4, 3, 475, 475)
    model = resnet50(pretrained=True)
    model.eval()
    output = model(input)
    print(output.shape)
