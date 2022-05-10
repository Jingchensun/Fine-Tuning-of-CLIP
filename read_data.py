# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import clip



chest_classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, mode='test', transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        self.mode = mode
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                if self.mode == 'test':
                    label = items[1:]
                    label = [int(i) for i in label]
                    labels.append(label)
                else:
                    label = items[1:]
                    label = [int(i) for i in label]
                    #print(label)
                    #print(label.type)
                    #label = np.array(label)
                    if 1 in label:
                        labels.append('a photo of a {}.'.format(chest_classes[label.index(1)]))
                        #print(label.index(1))
                        #print(labels)
                    else:
                        labels.append('a photo of normal.')

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        #print(index)
        if self.transform is not None:
            image = self.transform(image)
        if self.mode == 'test':
            return image, torch.FloatTensor(label)
        else:
            return image, label

# model, preprocess = clip.load("ViT-B/32")

# test_dataset = ChestXrayDataSet(data_dir='/home/jason/data/chestx_ray14/images/images',
#                                 image_list_file='/home/jason/CheXNet/ChestX-ray14/labels/test_list.txt',
#                                 mode = 'train',
#                                 transform=preprocess
#                                 # transform=transforms.Compose([
#                                 #     transforms.Resize(256),
#                                 #     transforms.TenCrop(224),
#                                     # ])
#                                     )
# test_loader = DataLoader(dataset=test_dataset, batch_size=8,
#                              shuffle=False, num_workers=8, pin_memory=True)

# for i, (images, target) in enumerate(tqdm(test_loader)):
#     print(images.size())
#     print(len(target))