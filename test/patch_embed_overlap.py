# -*- coding: utf-8 -*-
# ModuleName: patch_embed_overlap
# Author: dongyihua543
# Time: 2024/1/3 11:08

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.bases import ImageDataset
from datasets.market1501 import Market1501
from datasets.sampler import RandomIdentitySampler
from timm.data.random_erasing import RandomErasing
from model.backbones.vit_pytorch import PatchEmbed_overlap

# transform
train_transforms = T.Compose([
    T.Resize([256, 128], interpolation=3),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
    # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
])


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


if __name__ == '__main__':
    # 1. 数据集
    dataset = Market1501(root='../data/')
    train_set = ImageDataset(dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set, batch_size=64,
        sampler=RandomIdentitySampler(dataset.train, 64, 4),
        num_workers=1, collate_fn=train_collate_fn
    )

    # 2. Image to Patch Embedding with overlapping patches
    patch_embed = PatchEmbed_overlap(img_size=[256, 128], patch_size=16, stride_size=[12, 12], in_chans=3, embed_dim=768)

    # for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
    #     x = patch_embed(img)
    #     print(x.shape)

    # 迭代器
    train_loader_iterator = iter(train_loader)
    img = next(train_loader_iterator)[0]
    x = patch_embed(img)
    print(img.shape, '->', x.shape)
