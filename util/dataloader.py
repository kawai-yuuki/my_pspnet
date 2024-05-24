# util.dataloader.py
### ライブラリ
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torchvision import transforms


### クラス定義
class Compose(object):
    """
    指定した前処理を順次適用していくクラス
    
    Args:
        transforms (List[Transform]): 変換処理を格納したリスト
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno):
        for t in self.transforms:
            img, anno = t(img, anno)
        return img, anno

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    
class Resize(object):
    """
    指定したサイズにリサイズする
    """

    def __init__(self, size=475):
        self.size = size

    def __call__(self, image, anno_img):
        image = image.resize((self.size, self.size), Image.BICUBIC)
        anno_img = anno_img.resize((self.size, self.size), Image.NEAREST)
        return image, anno_img

    
class Normalize(object):
    """
    画像データを0～1に正規化する
    """
    def __init__(self):
        pass

    def __call__(self, image, anno_img):
        # 画像データをPILからTensorに変換
        image = transforms.functional.to_tensor(image)
        
        # 0～1に正規化
        image = image / 255.
        
        # アノテーション画像をNumpyに変換する
        anno_img = np.array(anno_img)
        
        # 境界値である255を0(backgroud)に変換する
        index = np.where(anno_img == 255)
        anno_img[index] = 0
        
        # アノテーション画像をTensorに変換する
        anno_img = torch.from_numpy(anno_img)

        return image, anno_img

    
class RandomVerticalFlip(object):
    """
    ランダムに画像の上下を反転させる
    """

    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img, anno_img):
        
        if torch.rand(1) < self.p:
            return transforms.functional.vflip(img), transforms.functional.vflip(anno_img)
        
        return img, anno_img
    
    
class RandomHorizontalFlip(object):
    """
    ランダムに画像の左右を反転させる
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, anno_img):
        
        if torch.rand(1) < self.p:
            return transforms.functional.hflip(img), transforms.functional.hflip(anno_img)
        
        return img, anno_img
    
    
class DataTransform(object):
    """
    画像データとアノテーションデータの前処理クラス。
    訓練時はデータ拡張を実施し、推論時はデータ拡張を実施しない。
    
    Attributes:
        input_size(int): リサイズ先の画像の大きさ
    """
    def __init__(self, input_size):
        self.data_transform = {
            'train': Compose([
                Resize(input_size),
                Normalize(),
                RandomHorizontalFlip(),
                RandomVerticalFlip()
            ]),
            'valid': Compose([
                Resize(input_size),
                Normalize()
            ])
        }
    
    def __call__(self, phase, img, anno_img):
        return self.data_transform[phase](img, anno_img)
    
    
class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス
    
    Attributes:
        img_list(list): 画像データのファイルパスを格納したリスト
        anno_img_list(list): アノテーションデータのファイルパスを格納したリスト
        phase(str): 'train' or 'valid'
        transform(object): 前処理クラスのインスタンス
    """
    
    def __init__(self, img_list, anno_img_list, phase, transform):
        self.img_list = img_list
        self.anno_img_list = anno_img_list
        self.phase = phase
        self.transform = transform
        
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.img_list)
    
    def __getitem__(self, index):
        """
        前処理後のTensor形式の画像データとアノテーションデータを返す
        """
        img, anno_img = self.pull_item(index)
        return img, anno_img

    def pull_item(self, index):
        """
        画像データとアノテーションデータを読み込み、前処理を実施する
        """
        # 読み込み
        img_file_path = self.img_list[index]
        anno_img_file_path = self.anno_img_list[index]
        
        img = Image.open(img_file_path)
        anno_img = Image.open(anno_img_file_path)
        
        # 前処理の実施
        img, anno_img = self.transform(self.phase, img, anno_img)
        
        return img, anno_img
    

### 関数定義
def make_datapath_list(rootpath):
    """
    学習用と検証用の画像データとアノテーションデータのファイルパスを格納したリストを取得する
    
    Args:
        rootpath(str): データフォルダへのパス
    
    Returns:
        train_img_list(list): 学習用の画像データへのファイルパス
        train_anno_list(list): 学習用のアノテーションデータへのファイルパス
        valid_img_list(list): 検証用の画像データへのファイルパス
        valid_anno_list(list): 検証用のアノテーションデータへのファイルパス
    """
    
    # 学習用画像の一覧を取得
    with open(os.path.join(rootpath, 'ImageSets/Segmentation/train.txt'), mode='r') as f:
        train_list = f.readlines()
        train_list = [val.replace('\n', '') for val in train_list]

    # 検証用画像の一覧を取得
    with open(os.path.join(rootpath, 'ImageSets/Segmentation/val.txt'), mode='r') as f:
        valid_list = f.readlines()
        valid_list = [val.replace('\n', '') for val in valid_list]
    
    # 学習用データのリストを作成
    train_img_list = [os.path.join(rootpath, f'JPEGImages/{val}.jpg') for val in train_list]
    train_anno_list = [os.path.join(rootpath, f'SegmentationClass/{val}.png') for val in train_list]
    
    # 検証用データのリストを作成
    valid_img_list = [os.path.join(rootpath, f'JPEGImages/{val}.jpg') for val in valid_list]
    valid_anno_list = [os.path.join(rootpath, f'SegmentationClass/{val}.png') for val in valid_list]
    
    return train_img_list, train_anno_list, valid_img_list, valid_anno_list


# 動作確認
if __name__ == '__main__':
    
    ### 1. ファイルパスのリストを作成

    # 学習用と検証用の画像データとアノテーションデータのファイルパスを格納したリストを取得する
    rootpath = '../data/VOCdevkit/VOC2012/'

    train_img_list, train_anno_list, valid_img_list, valid_anno_list = make_datapath_list(rootpath)

    print(train_img_list[0])
    print(train_anno_list[0])


    ### 2. 前処理クラスの作成

    # 稼働確認として画像ファイルを、サンプルとして読み込み
    train_img_path = train_img_list[0]
    train_anno_img_path = train_anno_list[0]

    img = Image.open(train_img_path)
    anno_img = Image.open(train_anno_img_path)

    # 前処理の実行
    transformer = DataTransform(input_size=475)
    img, anno_img = transformer('train', img, anno_img)

    print(img.shape)
    print(anno_img.shape)

    img_array = np.array(img) * 255
    img_array = img_array.transpose(1, 2, 0)

    anno_img_array = np.array(anno_img)

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(img_array)
    ax2.imshow(anno_img_array)

    plt.show()


    # 3. Datasetクラスの作成

    # 学習データのDataset
    train_dataset = VOCDataset(
        img_list=train_img_list,
        anno_img_list=train_anno_list,
        phase='train',
        transform=DataTransform(input_size=475)
    )

    # 検証データのDataset
    valid_dataset = VOCDataset(
        img_list=valid_img_list,
        anno_img_list=valid_anno_list,
        phase='valid',
        transform=DataTransform(input_size=475)
    )

    # データセットの取り出し
    print(valid_dataset.__getitem__(0)[0].shape)
    print(valid_dataset.__getitem__(0)[1].shape)


    ### 4. DataLoaderの作成

    batch_size = 8

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

    # 動作確認
    batch_iterator = iter(valid_dataloader)
    imgs, anno_imgs = next(batch_iterator)

    print(imgs.shape)
    print(anno_imgs.shape)