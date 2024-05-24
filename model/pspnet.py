# model.pspnet.py
### ライブラリ
import sys

from torch.nn.modules import padding
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import resnet50


### クラス定義
class PyramidPoolingModule(nn.Module):
    def __init__(self):
        super(PyramidPoolingModule, self).__init__()
        
        # 1×1スケール
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

        # 2×2スケール
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=2)
        self.conv2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 3×3スケール
        self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=3)
        self.conv3 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 4×4スケール
        self.avgpool6 = nn.AdaptiveAvgPool2d(output_size=6)
        self.conv6 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 1×1スケール
        out1 = self.avgpool1(x)
        out1 = self.conv1(out1)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = F.interpolate(out1, (60, 60), mode='bilinear', align_corners=True)
        
        # 2×2スケール
        out2 = self.avgpool2(x)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        out2 = F.interpolate(out2, (60, 60), mode='bilinear', align_corners=True)
        
        # 3×3スケール
        out3 = self.avgpool3(x)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        out3 = F.interpolate(out3, (60, 60), mode='bilinear', align_corners=True)
        
        # 6×6スケール
        out6 = self.avgpool6(x)
        out6 = self.conv6(out6)
        out6 = self.bn6(out6)
        out6 = self.relu6(out6)
        out6 = F.interpolate(out6, (60, 60), mode='bilinear', align_corners=True)
        
        # 元の入力と各スケールの特徴量を結合させる
        out = torch.cat([x, out1, out2, out3, out6], dim=1)
        
        return out


class PSPNet(nn.Module):
    
    def __init__(self, n_classes=21):
        super(PSPNet, self).__init__()
        self.n_classes = n_classes

        # ResNetから最初の畳み込みフィルタとlayer1からlayer4を取得する
        resnet = resnet50(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, 
            resnet.conv2, resnet.bn2, resnet.relu, 
            resnet.conv3, resnet.bn3, resnet.relu, 
            resnet.maxpool
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # layer3とlayer4の畳み込みフィルタのパラメータを変更する
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)

            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        # Pyramid Pooling モジュール
        self.ppm = PyramidPoolingModule()

        # UpSampling モジュール
        self.cls = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, self.n_classes, kernel_size=1)
        )

        # 学習時にのみAuxLossモジュールを使用するように設定
        if self.training is True:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, self.n_classes, kernel_size=1)
            )
            
    def forward(self, x, y=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # AuxLossのためにlayer3から出力を抜き出しておく
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        # Pyramid Pooling モジュール
        x = self.ppm(x)

        # UpSampling モジュール
        x = self.cls(x)
        # 入力画像と同じ大きさに変換する
        x = F.interpolate(x, size=(475, 475), mode='bilinear', align_corners=True)

        # 学習時にのみAuxLossモジュールを使用するように設定
        if self.training is True:
            aux = self.aux(x_tmp)
            aux = F.interpolate(aux, size=(475, 475), mode='bilinear', align_corners=True)
            return x, aux

        return x


# 動作確認
if __name__ == '__main__':
    input = torch.rand(4, 3, 475, 475)
    model = PSPNet()

    # モデルを学習用に設定した場合
    model.train()
    output, aux = model(input)
    print('学習時')
    print(output.shape)
    print(aux.shape)

    # モデルを推論用に設定した場合
    model.eval()
    output = model(input)
    print('推論時')
    print(output.shape)