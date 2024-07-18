'''
@anthor: Wenyuan Li
@desc: Datasets for self-supervised
@date: 2020/5/15
'''
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision
import glob
import os
import numpy as np
from skimage import io
from skimage import util as sk_utils
from PIL import Image
from SSLRemoteSensing.datasets.transforms.representation import inpainting_transforms,builder
import torch.utils.data as data_utils
from SSLRemoteSensing.datasets.transforms.representation.agr_transforms import AGRTransforms

# class InpaintingAGRDataset(data_utils.Dataset):

#     def __init__(self,data_path,data_format,inpainting_transforms_cfg,agr_transforms_cfg,
#                  pre_transforms_cfg,post_transforms_cfg, img_size=256):
#         super(InpaintingAGRDataset, self).__init__()

#         self.data_files=glob.glob(os.path.join(data_path,data_format))

#         self.img_size=img_size
#         self.inpainting_transforms=inpainting_transforms.InpaintingTransforms(**inpainting_transforms_cfg)
#         pre_transforms=[]
#         for param in pre_transforms_cfg.values():
#             pre_transforms.append(builder.build_transforms(**param))
#         self.pre_transforms=torchvision.transforms.Compose(pre_transforms)

#         post_transforms=[]
#         for param in post_transforms_cfg.values():
#             post_transforms.append(builder.build_transforms(**param))
#         self.post_transforms=torchvision.transforms.Compose(post_transforms)

#         self.agr_transforms =AGRTransforms(**agr_transforms_cfg)


#     def __getitem__(self, item):

#         img=Image.open(self.data_files[item])
#         img=self.pre_transforms(img)

#         inpainting_label=img

#         pre_img=img
#         post_img,agr_label=self.agr_transforms.forward(img)
#         data=img
#         data=self.inpainting_transforms(data)
#         data=self.post_transforms(data)
#         inpainting_label=self.post_transforms(inpainting_label)
#         inpainting_mask=torch.abs(inpainting_label-data)
#         pre_img=self.post_transforms(pre_img)
#         post_img=self.post_transforms(post_img)
#         agr_label=torch.tensor(agr_label,dtype=torch.int64)
#         return data,pre_img,post_img, inpainting_label,inpainting_mask,agr_label

#     def __len__(self):
#         return len(self.data_files)

# class LabeledInpaintingAGRDataset(data_utils.Dataset):

#     def __init__(self, data_path, inpainting_transforms_cfg, agr_transforms_cfg,
#                  pre_transforms_cfg, post_transforms_cfg, img_size=256):
#         super(LabeledInpaintingAGRDataset, self).__init__()

#         self.data_path = data_path
#         self.labels = os.listdir(data_path)
#         self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
#         self.img_size = img_size
#         self.inpainting_transforms = inpainting_transforms.InpaintingTransforms(**inpainting_transforms_cfg)

#         pre_transforms = [builder.build_transforms(**param) for param in pre_transforms_cfg.values()]
#         self.pre_transforms = transforms.Compose(pre_transforms)

#         post_transforms = [builder.build_transforms(**param) for param in post_transforms_cfg.values()]
#         self.post_transforms = transforms.Compose(post_transforms)

#         self.agr_transforms = AGRTransforms(**agr_transforms_cfg)

#         # Create a list of all image paths and corresponding labels
#         self.image_paths = []
#         self.image_labels = []
#         for label in self.labels:
#             sample_paths = os.listdir(os.path.join(self.data_path, label))
#             for sample in sample_paths:
#                 img_paths = glob.glob(os.path.join(self.data_path, label, sample, '*.tif'))
#                 self.image_paths.extend(img_paths)
#                 self.image_labels.extend([self.label_to_index[label]] * len(img_paths))

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         label = self.image_labels[index]

#         # Load and process the image
#         img = Image.open(img_path)
#         img = img.convert('RGB')
#         img = self.pre_transforms(img)

#         inpainting_label = img
#         pre_img = img
#         post_img, agr_label = self.agr_transforms.forward(img)
#         data = img
#         data = self.inpainting_transforms(data)
#         data = self.post_transforms(data)
#         inpainting_label = self.post_transforms(inpainting_label)
#         inpainting_mask = torch.abs(inpainting_label - data)
#         pre_img = self.post_transforms(pre_img)
#         post_img = self.post_transforms(post_img)
#         agr_label = torch.tensor(agr_label, dtype=torch.int64)

#         return data, pre_img, post_img, inpainting_label, inpainting_mask, agr_label, label

#     def __len__(self):
#         return len(self.image_paths)
class InpaintingAGRDataset(data_utils.Dataset):

    def __init__(self, data_path, inpainting_transforms_cfg, agr_transforms_cfg,
                 pre_transforms_cfg, post_transforms_cfg, img_size=256):
        super(InpaintingAGRDataset, self).__init__()

        self.data_path = data_path
        self.img_size = img_size
        self.inpainting_transforms = inpainting_transforms.InpaintingTransforms(**inpainting_transforms_cfg)

        pre_transforms = [builder.build_transforms(**param) for param in pre_transforms_cfg.values()]
        self.pre_transforms = transforms.Compose([transforms.Resize((self.img_size, self.img_size))] + pre_transforms)

        post_transforms = [builder.build_transforms(**param) for param in post_transforms_cfg.values()]
        self.post_transforms = transforms.Compose([transforms.Resize((self.img_size, self.img_size))] + post_transforms)

        self.agr_transforms = AGRTransforms (**agr_transforms_cfg)

        # Add ToTensor and Resize transforms
        self.to_tensor = transforms.ToTensor()

        # Create a list of all image paths
        self.image_paths = []
        sample_names = os.listdir(self.data_path)
        for sample_name in sample_names:
            smaller_sample_paths = os.listdir(os.path.join(self.data_path, sample_name))
            for smaller_sample in smaller_sample_paths:
                img_paths = glob.glob(os.path.join(self.data_path, sample_name, smaller_sample, '*.tif'))
                self.image_paths.append(img_paths)  # Store the list of paths for each smaller sample

    def __getitem__(self, index):
        img_paths = self.image_paths[index]

        # Load and stack the 12 .tif images for this sample
        imgs = [Image.open(p).convert('RGB') for p in img_paths]
        imgs = [self.pre_transforms(img) for img in imgs]

        inpainting_label = [self.post_transforms(img) for img in imgs]
        pre_img = [self.post_transforms(img) for img in imgs]
        post_img_agr_label = [self.agr_transforms.forward(img) for img in imgs]

        post_img = []
        agr_label = []
        for idx in range(len(post_img_agr_label)):
            post_img.append(self.post_transforms(post_img_agr_label[idx][0]))
            agr_label.append(post_img_agr_label[idx][1])

        data = [self.post_transforms(self.inpainting_transforms(img)) for img in imgs]
        inpainting_mask = [torch.abs(inpainting_label[idx] - data[idx]) for idx in range(len(data))]

        agr_label = torch.tensor(agr_label, dtype=torch.int64)
        inpainting_label = torch.cat(inpainting_label, dim=0)  # Concatenate along channel dimension
        pre_img = torch.cat(pre_img, dim=0)  # Concatenate along channel dimension
        post_img = torch.cat(post_img, dim=0)  # Concatenate along channel dimension
        data = torch.cat(data, dim=0)  # Concatenate along channel dimension
        inpainting_mask = torch.cat(inpainting_mask, dim=0)  # Concatenate along channel dimension
        
        return data, pre_img, post_img, inpainting_label, inpainting_mask, agr_label

    def __len__(self):
        return len(self.image_paths)
    
class LabeledInpaintingAGRDataset(data_utils.Dataset):

    def __init__(self, data_path, inpainting_transforms_cfg, agr_transforms_cfg,
                 pre_transforms_cfg, post_transforms_cfg, img_size=256):
        super(LabeledInpaintingAGRDataset, self).__init__()

        self.data_path = data_path
        self.labels = os.listdir(data_path)
        self.labels.pop(2) # remove the evaluation
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.img_size = img_size
        self.inpainting_transforms = inpainting_transforms.InpaintingTransforms(**inpainting_transforms_cfg)

        pre_transforms = [builder.build_transforms(**param) for param in pre_transforms_cfg.values()]
        self.pre_transforms = transforms.Compose([transforms.Resize((self.img_size, self.img_size))] + pre_transforms)

        post_transforms = [builder.build_transforms(**param) for param in post_transforms_cfg.values()]
        self.post_transforms = transforms.Compose([transforms.Resize((self.img_size, self.img_size))] + post_transforms)

        self.agr_transforms = AGRTransforms(**agr_transforms_cfg)

        # Add ToTensor and Resize transforms
        self.to_tensor = transforms.ToTensor()

        # Create a list of all image paths and corresponding labels
        self.image_paths = []
        self.image_labels = []
        for label in self.labels:
            sample_paths = os.listdir(os.path.join(self.data_path, label))
            for sample in sample_paths:
                img_paths = glob.glob(os.path.join(self.data_path, label, sample, '*.tif'))
                self.image_paths.extend([img_paths])  # Store the list of paths for each sample
                self.image_labels.append(self.label_to_index[label])

    def __getitem__(self, index):
        img_paths = self.image_paths[index]
        label = self.image_labels[index]

        # Load and stack the 12 .tif images for this sample
        imgs = [Image.open(p).convert('RGB') for p in img_paths]
        imgs = [self.pre_transforms(img) for img in imgs]

        inpainting_label = [self.post_transforms(img) for img in imgs]
        pre_img = [self.post_transforms(img) for img in imgs]
        post_img_agr_label = [self.agr_transforms.forward(img) for img in imgs]

        post_img = []
        agr_label = []
        for idx in range(len(post_img_agr_label)):
            post_img.append(self.post_transforms(post_img_agr_label[idx][0]))
            agr_label.append(post_img_agr_label[idx][1])

        data = [self.post_transforms(self.inpainting_transforms(img)) for img in imgs]
        inpainting_mask = [torch.abs(inpainting_label[idx] - data[idx]) for idx in range(len(data))]
              
        agr_label = torch.tensor(agr_label, dtype=torch.int64)
        inpainting_label = torch.cat(inpainting_label, dim=0)  # Concatenate along channel dimension
        pre_img = torch.cat(pre_img, dim=0)  # Concatenate along channel dimension
        post_img = torch.cat(post_img, dim=0)  # Concatenate along channel dimension
        data = torch.cat(data, dim=0)  # Concatenate along channel dimension
        inpainting_mask = torch.cat(inpainting_mask, dim=0)  # Concatenate along channel dimension
        
        return data, pre_img, post_img, inpainting_label, inpainting_mask, agr_label, label

    def __len__(self):
        return len(self.image_paths)