# notebool/01_ファインチューニング_準備偏.ipynb
### ライブラリ
import sys
sys.path.append('../')

import torch
from model.pspnet import PSPNet

from collections import OrderedDict

### 中身の確認
# 実装したPSPNetのKey名を確認する
model = PSPNet(n_classes=150)

print('実装したPSPNet')
print('Keyの数 : ', len(model.state_dict().keys()))
print('Key名 : ')
print(model.state_dict().keys())

# ダウンロードしたPSPNetのKey名を確認する
param_path = '../parameters/trained_model/train_epoch_100.pth'
param_ade = torch.load(param_path, map_location=torch.device('cpu'))

print('ダウンロードしたPSPNet')
print('Keyの数 : ', len(param_ade['state_dict'].keys()))
print('Key名')
print(param_ade['state_dict'].keys())

# Key名が一致しないとエラーが出力される
# model.load_state_dict(param_ade)

### Key名の変更
# パラメータ内のキーを変更
param_list = []

ppm_name_list = [
    'conv1', 'bn1', 'bn1', 'bn1', 'bn1', 'bn1',
    'conv2', 'bn2', 'bn2', 'bn2', 'bn2', 'bn2',
    'conv3', 'bn3', 'bn3', 'bn3', 'bn3', 'bn3',
    'conv6', 'bn6', 'bn6', 'bn6', 'bn6', 'bn6',
]

for before_key in param_ade['state_dict'].keys():
    after_key = before_key.replace('module.', '').replace('features.', '')
    
    if 'ppm' in after_key:
        after_key = after_key.replace('ppm.', 'ppm.{}.'.format(ppm_name_list.pop(0)))
        after_key = after_key.replace('.0', '').replace('.1', '').replace('.2', '').replace('.3', '')

    param_list.append((after_key, param_ade['state_dict'][before_key]))
    
param_ade_rename = OrderedDict(param_list)

### 変更結果の確認
model = PSPNet(n_classes=150)
model.load_state_dict(param_ade_rename)

### モデルの保存
torch.save(model.state_dict(), '../parameters/trained_model/pspnet_base.pth')
