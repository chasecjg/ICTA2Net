import sys, os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

from os import path
d = path.dirname(__file__)
parent_path = os.path.dirname(d)
sys.path.append(parent_path)
import pandas as pd
import os
import cv2

from itertools import combinations
from PIL import Image
import torchvision.transforms as transforms
import random
import torch
import argparse
from typing import List, Union
from torch.utils.data import Dataset, DataLoader




IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

from DETRIS.utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

import random
# 设置随机种子
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result





class AVADataset(Dataset):
    def __init__(self, path_to_csv, root_dir, if_train=True, word_length=77, ablate_text=False):
        self.ablate_text = ablate_text
        # 读取CSV文件
        self.df = pd.read_csv(path_to_csv)
        self.root_dir = root_dir
        self.word_length = word_length

        # 预生成所有图像对
        # Group data by FileName for efficient sampling
        self.groups = self.df.groupby("FileName")

        # 为每个FileName分配唯一的数字组ID
        self.file_name_to_group_id = {name: idx for idx, name in enumerate(self.groups.groups.keys())}

        # 为每一组图像生成所有两两组合
        self.pairs = []
        for file_name, group_data in self.groups:
            group_id = self.file_name_to_group_id[file_name]
            # 找到 gt 图像
            gt_image = group_data[(group_data["Temperature_class"] == 0) & (group_data["Flag"] == 99999.0)]
            if not gt_image.empty:
                gt_index = gt_image.index[0]
                other_indices = group_data.index.drop(gt_index)
                # gt 与其他图像组合
                for idx in other_indices:
                    self.pairs.append((gt_index, idx, group_id))

            # 标签为 -1 和 1 的图像自身组合
            minus_one_images = group_data[group_data["Temperature_class"] == -1]
            plus_one_images = group_data[group_data["Temperature_class"] == 1]
            if len(minus_one_images) > 1:
                minus_one_combinations = list(combinations(minus_one_images.index, 2))
                # self.pairs.extend(minus_one_combinations)
                self.pairs.extend([(i, j, group_id) for i, j in minus_one_combinations])
            if len(plus_one_images) > 1:
                plus_one_combinations = list(combinations(plus_one_images.index, 2))
                # self.pairs.extend(plus_one_combinations)
                self.pairs.extend([(i, j, group_id) for i, j in plus_one_combinations])

        # print(f"Total number of pairs: {len(self.pairs)}")
        # print(self.pairs)
        # 随机选择一半的图像对交换顺序
        num_pairs_to_swap = len(self.pairs) // 2
        indices_to_swap = random.sample(range(len(self.pairs)), num_pairs_to_swap)
        for idx in indices_to_swap:
            i, j, gid = self.pairs[idx]
            self.pairs[idx] = (j, i, gid)  # 保持组ID不变
            # self.pairs[idx] = (self.pairs[idx][1], self.pairs[idx][0])
        # 定义数据变换
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        # 数据集的长度就是所有图像对的数量
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Args:
            index (int): 索引，用于从所有图像对中取一对。

        Returns:
            img1 (Tensor): 第一个图像的tensor。
            img2 (Tensor): 第二个图像的tensor。
            text1 (str): 第一个图像的描述信息。
            text2 (str): 第二个图像的描述信息。
            score1 (float): 第一个图像的score。
            score2 (float): 第二个图像的score。
            label (int): 图像对的标签，根据 new_score 判断
        """
        # 获取对应的图像对索引
        idx1, idx2, group_id = self.pairs[index]

        # 获取这两个图像对应的行
        sample1 = self.df.loc[idx1]
        sample2 = self.df.loc[idx2]

        # 获取图像路径
        img1_path = os.path.join(self.root_dir, sample1["FileName"], sample1["ImageName"])
        img2_path = os.path.join(self.root_dir, sample2["FileName"], sample2["ImageName"])
        # 使用OpenCV读取图像
        img1 = cv2.imread(img1_path)  # 读取图像
        img2 = cv2.imread(img2_path)  # 读取图像

        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Image not found: {img1_path if img1 is None else img2_path}")

        # 转换BGR为RGB格式
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Apply transforms if specified
        img1 = self.transform(Image.fromarray(img1))  # 转换为PIL图像后应用transform
        img2 = self.transform(Image.fromarray(img2))  # 转换为PIL图像后应用transform

        # Extract deltaE2000 scores
        # text1 = sample1["Description"]
        # text2 = sample2["Description"]
        # if self.ablate_text:
        #     text1_vec = torch.zeros(self.word_length).int()  # 设定一个零向量
        #     text2_vec = torch.zeros(self.word_length).int()  # 设定一个零向量            
        # else:  # Add else clause to handle the case when ablate_text is False
        #     # 对文本进行分词、编码和长度处理
        #     text1_vec = tokenize(text1, self.word_length, True).squeeze(0)
        #     text2_vec = tokenize(text2, self.word_length, True).squeeze(0)

        if self.ablate_text:
            text1 = ""
            text2 = ""
        else:
            text1 = sample1["Description"]
            text2 = sample2["Description"]

        text1_vec = tokenize(text1, self.word_length, True).squeeze(0)
        text2_vec = tokenize(text2, self.word_length, True).squeeze(0)
        score1 = sample1["new_score"]
        score2 = sample2["new_score"]

        # 判断标签
        if score1 > score2:
            label = 1
        elif score1 < score2:
            label = -1
        else:
            label = 0
        # print("*"*100)
        # print(img1_path)
        # print(img2_path)
        # # print(sample1["Temperature_class"])
        # # print(sample2["Temperature_class"])
        # print(text1)
        # print(score1)
        # print(score2)
        # print(label)
        # print(group_id)
        return img1, img2, text1_vec, text2_vec, score1, score2, label, sample1["ImageName"], sample2["ImageName"], group_id




class AVADataset_test(Dataset):
    def __init__(self, path_to_csv, root_dir, if_train=True, word_length=77, ablate_text=False):
        self.ablate_text = ablate_text
        # 读取CSV文件
        self.df = pd.read_csv(path_to_csv)
        self.root_dir = root_dir
        self.word_length = word_length
        
        # 预生成所有图像对
        # Group data by ImageName for efficient sampling
        # self.groups = self.df.groupby("FileName")
        self.groups = self.df.groupby("pair")
        
        # 建立FileName到group_id的映射
        self.file_name_to_group_id = {}
        current_id = 0
        for file_name in self.df["FileName"].unique():
            self.file_name_to_group_id[file_name] = current_id
            current_id += 1


        # 为每一组图像生成所有两两组合
        self.pairs = []
        for _, group_data in self.groups:
            group_combinations = list(combinations(group_data.index, 2))  # 获取所有的图像对组合
            self.pairs.extend(group_combinations)

        # 随机选择一半的图像对交换顺序
        num_pairs_to_swap = len(self.pairs) // 2
        indices_to_swap = random.sample(range(len(self.pairs)), num_pairs_to_swap)
        for idx in indices_to_swap:
            self.pairs[idx] = (self.pairs[idx][1], self.pairs[idx][0])
        # 定义数据变换
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        # 数据集的长度就是所有图像对的数量
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Args:
            index (int): 索引，用于从所有图像对中取一对。

        Returns:
            img1 (Tensor): 第一个图像的tensor。
            img2 (Tensor): 第二个图像的tensor。
            text1 (str): 第一个图像的描述信息。
            text2 (str): 第二个图像的描述信息。
            score1 (float): 第一个图像的score。
            score2 (float): 第二个图像的score。
        """
        # 获取对应的图像对索引
        idx1, idx2 = self.pairs[index]

        # 获取这两个图像对应的行
        sample1 = self.df.loc[idx1]
        sample2 = self.df.loc[idx2]

        # 获取图像路径
        img1_path = os.path.join(self.root_dir, sample1["FileName"], sample1["ImageName"])
        img2_path = os.path.join(self.root_dir, sample2["FileName"], sample2["ImageName"])
        # 使用OpenCV读取图像
        img1 = cv2.imread(img1_path)  # 读取图像
        img2 = cv2.imread(img2_path)  # 读取图像

        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Image not found: {img1_path if img1 is None else img2_path}")

        # 转换BGR为RGB格式
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Apply transforms if specified
        img1 = self.transform(Image.fromarray(img1))  # 转换为PIL图像后应用transform
        img2 = self.transform(Image.fromarray(img2))  # 转换为PIL图像后应用transform

        # # Extract deltaE2000 scores
        # text1 = sample1["Description"]
        # text2 = sample2["Description"]
        # if self.ablate_text:
        #     text1_vec = torch.zeros(self.word_length).int()  # 设定一个零向量
        #     text2_vec = torch.zeros(self.word_length).int()  # 设定一个零向量            
        # else:  # Add else clause to handle the case when ablate_text is False
        # # 对文本进行分词、编码和长度处理
        #     text1_vec = tokenize(text1, self.word_length, True).squeeze(0)
        #     text2_vec = tokenize(text2, self.word_length, True).squeeze(0)
        # Extract deltaE2000 scores

        if self.ablate_text:
            text1 = ""
            text2 = ""
        else:
            text1 = sample1["Description"]
            text2 = sample2["Description"]

        text1_vec = tokenize(text1, self.word_length, True).squeeze(0)
        text2_vec = tokenize(text2, self.word_length, True).squeeze(0)
        score1 = sample1["Majority_Label"]
        score2 = sample2["Majority_Label"]

        # 判断标签
        if score1 > score2:
            label = 1
        elif score1 < score2:
            label = -1
        else:
            label = 0

        confidence1 = sample1["Confidence"]
        group_id = self.file_name_to_group_id[sample1["FileName"]]

        # 打印输出路径和返回的三个变量
        # print("*"*100)
        # print(f"Image 1 Path: {img1_path}")
        # print(f"Image 2 Path: {img2_path}")
        # print(img1.shape)
        # print(f"Text for Image 1: {text1_vec.shape}")
        # print(f"Text for Image 2: {text2_vec.shape}")
        # print(text1_vec)
        # print(f"Score 1: {score1}")
        # print(f"Score 2: {score2}")
        # print(group_id)
        # print("-" * 30)
        return img1, img2, text1_vec, text2_vec, score1, score2, label, sample1["ImageName"], sample2["ImageName"], confidence1, group_id



class AVADataset_test_all_pair(Dataset):
    def __init__(self, path_to_csv, root_dir, if_train=True, word_length=77, ablate_text=False):
        self.ablate_text = ablate_text
        # 读取CSV文件
        self.df = pd.read_csv(path_to_csv)
        self.root_dir = root_dir
        self.word_length = word_length

        # 预生成所有图像对
        # Group data by FileName for efficient sampling
        self.groups = self.df.groupby("FileName")
        # 为每一组图像生成所有两两组合
        self.pairs = []
        for _, group_data in self.groups:
            group_combinations = list(combinations(group_data.index, 2))  # 获取所有的图像对组合
            self.pairs.extend(group_combinations)


        num_pairs_to_swap = len(self.pairs) // 2
        # print(f"num_pairs_to_swap: {num_pairs_to_swap}")
        indices_to_swap = random.sample(range(len(self.pairs)), num_pairs_to_swap)
        # print(f"indices_to_swap: {indices_to_swap}")
        for idx in indices_to_swap:
            self.pairs[idx] = (self.pairs[idx][1], self.pairs[idx][0])
            
        # 定义数据变换
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        # 数据集的长度就是所有图像对的数量
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Args:
            index (int): 索引，用于从所有图像对中取一对。

        Returns:
            img1 (Tensor): 第一个图像的tensor。
            img2 (Tensor): 第二个图像的tensor。
            text1 (str): 第一个图像的描述信息。
            text2 (str): 第二个图像的描述信息。
            score1 (float): 第一个图像的score。
            score2 (float): 第二个图像的score。
            label (int): 图像对的标签，根据 new_score 判断
        """
        # 获取对应的图像对索引
        idx1, idx2 = self.pairs[index]

        # 获取这两个图像对应的行
        sample1 = self.df.loc[idx1]
        sample2 = self.df.loc[idx2]

        # 获取图像路径
        img1_path = os.path.join(self.root_dir, sample1["FileName"], sample1["ImageName"])
        img2_path = os.path.join(self.root_dir, sample2["FileName"], sample2["ImageName"])
        # 使用OpenCV读取图像
        img1 = cv2.imread(img1_path)  # 读取图像
        img2 = cv2.imread(img2_path)  # 读取图像

        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Image not found: {img1_path if img1 is None else img2_path}")

        # 转换BGR为RGB格式
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Apply transforms if specified
        img1 = self.transform(Image.fromarray(img1))  # 转换为PIL图像后应用transform
        img2 = self.transform(Image.fromarray(img2))  # 转换为PIL图像后应用transform

        # Extract deltaE2000 scores
        # text1 = sample1["Description"]
        # text2 = sample2["Description"]
        # if self.ablate_text:
        #     text1_vec = torch.zeros(self.word_length).int()  # 设定一个零向量
        #     text2_vec = torch.zeros(self.word_length).int()  # 设定一个零向量            
        # else:  # Add else clause to handle the case when ablate_text is False
        #     # 对文本进行分词、编码和长度处理
        #     text1_vec = tokenize(text1, self.word_length, True).squeeze(0)
        #     text2_vec = tokenize(text2, self.word_length, True).squeeze(0)

        if self.ablate_text:
            text1 = ""
            text2 = ""
        else:
            text1 = sample1["Description"]
            text2 = sample2["Description"]

        text1_vec = tokenize(text1, self.word_length, True).squeeze(0)
        text2_vec = tokenize(text2, self.word_length, True).squeeze(0)
        score1 = sample1["new_score"]
        score2 = sample2["new_score"]

        # 判断标签
        if score1 > score2:
            label = 1
        elif score1 < score2:
            label = -1
        else:
            label = 0
        # print("*"*100)
        # print(img1_path)
        # print(img2_path)
        # print(sample1["Temperature_class"])
        # print(sample2["Temperature_class"])
        # print(text1)
        # print(score1)
        # print(score2)
        # print(label)
        return img1, img2, text1_vec, text2_vec, score1, score2, label, sample1["ImageName"], sample2["ImageName"]

# test example:
parser = argparse.ArgumentParser(description="Your program description here")
parser.add_argument('--path_to_save_csv', type=str, default='/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/data/', help='directory to csv')
parser.add_argument('--image_test_path', type=str,default="/mnt/group_temp/cjg/FiveK/group_image_filter_Tint_0_gt", help='directory to imag')
opt = parser.parse_args()

if __name__ == "__main__":
    train_csv_path = os.path.join(opt.path_to_save_csv, 'FiveK_test_0_1_conf.csv')
    print("train_csv_path:{}".format(train_csv_path))
    train_ds = AVADataset_test(train_csv_path, opt.image_test_path, if_train=True, ablate_text=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    
    # 遍历DataLoader，打印每条数据
    for i, (img1, img2, text1_vec, text2_vec, score1, score2, label,_,_,_, group_id) in enumerate(train_loader):
        pass
#         # print("*"*100)
        # print(img1.shape)
        # print(img2.shape)
        # print(f"Text for Image 1: {text1_vec.shape}")
        # print(f"Text for Image 2: {text2_vec.shape}")
        # print("score1:{}".format(score1))
        # print("score2:{}".format(score2))
        # print("label:{}".format(label))
        # print("-" * 30)
