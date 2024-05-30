## ライブラリ
import sys
sys.path.append('../')

import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import CSVLogger

from util.dataloader import make_datapath_list, DataTransform, VOCDataset
from model.pspnet import PSPNet

import matplotlib.pyplot as plt

## 1. モデル定義
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()

        # PSPNetの定義
        # 学習済みモデルの読み込み
        model = PSPNet(n_classes=150)
        model_path = '../parameters/trained_model/pspnet_base.pth'
        model.load_state_dict(torch.load(model_path))
        # VOC2012にあうように変更
        model.cls[4] = nn.Conv2d(512, 21, kernel_size=1)
        model.aux[4] = nn.Conv2d(256, 21, kernel_size=1)
        # パラメータの初期化
        model.cls[4].apply(self.weights_init)
        model.aux[4].apply(self.weights_init)
        self.model = model

        # 損失関数の定義
        # 損失関数にはクロスエントロピーを使用
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # ファインチューニングなので、学習率は小さめに設定
        optimizer = optim.SGD([
            {'params': self.model.layer0.parameters(), 'lr': 1e-3},
            {'params': self.model.layer1.parameters(), 'lr': 1e-3},
            {'params': self.model.layer2.parameters(), 'lr': 1e-3},
            {'params': self.model.layer3.parameters(), 'lr': 1e-3},
            {'params': self.model.layer4.parameters(), 'lr': 1e-3},
            {'params': self.model.ppm.parameters(), 'lr': 1e-3},
            {'params': self.model.cls.parameters(), 'lr': 1e-2},
            {'params': self.model.aux.parameters(), 'lr': 1e-2},
        ], momentum=0.9, weight_decay=0.0001)

        return optimizer

    def training_step(self, batch, batch_idx):
        # optimizerの取得
        optimizer = self.optimizers()
        optimizer.zero_grad()

        # 損失の計算
        images, targets = batch
        preds = self(images)
        main_loss = self.criterion(preds[0], targets.long())
        aux_loss = self.criterion(preds[1], targets.long())
        loss = main_loss + aux_loss * 0.4

        # ログに保存
        self.log('train_main_loss', main_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('train_aux_loss', aux_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # 損失の計算
        # 推論時はAuxLossモジュールからの出力はないことに注意
        images, targets = batch
        preds = self(images)
        loss = self.criterion(preds, targets.long())
        
        # ログに保存
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
    
            if m.bias is not None:  # バイアス項がある場合
                nn.init.constant_(m.bias, 0.0)

## 2. 学習の設定
# CheckPointの設定
checkpoint = callbacks.ModelCheckpoint(
    dirpath='../parameters/checkpoint/',
    # 5 epochごとにモデルを保存する
    every_n_epochs=5,
    save_top_k=-1
)

# Loggerの設定
logger = CSVLogger(
    '../parameters/checkpoint/',
    name='pspnet'
)

# Trainerの設定
trainier = pl.Trainer(
    max_epochs=30,
    accelerator='gpu',
    accumulate_grad_batches=8,
    callbacks=[checkpoint],
    logger=logger,
)

## 3. 学習の実行
## DataLoaderの作成
# 学習用と検証用の画像データとアノテーションデータのファイルパスを格納したリストを取得
rootpath = '../data/VOCdevkit/VOC2012/'
train_img_list, train_anno_list, valid_img_list, valid_anno_list = make_datapath_list(rootpath)

# Datasetの作成
train_dataset = VOCDataset(
    img_list=train_img_list,
    anno_img_list=train_anno_list,
    phase='train',
    transform=DataTransform(input_size=475)
)
valid_dataset = VOCDataset(
    img_list=valid_img_list,
    anno_img_list=valid_anno_list,
    phase='valid',
    transform=DataTransform(input_size=475)
)

# DataLoaderの作成
batch_size = 4

# 学習データのDataLoader
train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# 検証データのDataLoader
valid_dataloader = data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False
)

## 学習の実行
model = Model()

trainier.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)

## 4. 推論
# 推論対象画像の取得と変換
image_name = '2007_000033'
image = Image.open(os.path.join('../data/VOCdevkit/VOC2012/JPEGImages/', f'{image_name}.jpg'))
anno_image = Image.open(os.path.join('../data/VOCdevkit/VOC2012/SegmentationClass/', f'{image_name}.png'))

transform = DataTransform(input_size=475)
transformed_image, transformerd_anno = transform('valid', image, anno_image)
transformed_image = transformed_image.unsqueeze(0)

# モデルの読み込み
model = Model.load_from_checkpoint('../parameters/checkpoint/epoch=29-step=1380.ckpt')
# 推論モードに設定
model.eval()

# 推論の実施と属するクラスの取得
pred_image = model(transformed_image.to('cuda:0'))
pred_image = pred_image.argmax(1)[0]
pred_image = pred_image.detach().to('cpu').numpy().astype(np.uint8)

# パレットモードに変換
p_palette = anno_image.getpalette()

pred_image_p = Image.fromarray(pred_image)
pred_image_p = pred_image_p.convert('P')
pred_image_p.putpalette(p_palette)

# 結果の確認
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.imshow(anno_image.resize((475, 475)))
ax2.imshow(pred_image_p)

plt.show()
