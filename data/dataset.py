from this import d
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import torch
import torchvision.transforms as transform
import torchvision.transforms.functional as F
import random

class MyRandomCrop(object):
    def __init__(self, max_crop_size) -> None:
        self.max_crop_size = max_crop_size
    def __call__(self, img):
        (h, w) = img.shape[-2:]
        crop_size = min(min(h, w), self.max_crop_size)
        top = random.randint(0, h-crop_size)
        left = random.randint(0, w-crop_size)
        return F.crop(img, top=top, left=left, height=crop_size, width=crop_size)

class MyResize(object):
    def __init__(self, image_width) -> None:
        self.image_width = image_width
    def __call__(self, img):
        (h, w) = img.shape[-2:]
        new_h = int(h * (self.image_width / w)) // 4 * 4
        new_w = self.image_width // 4 * 4
        return F.resize(img, (new_h, new_w), transform.InterpolationMode.BICUBIC)

class CycleGANDataset(Dataset):
    def __init__(self, domainA_path, name_filter_domainA, domainB_path, name_filter_domainB, image_size, crop_size, max_data_per_epoch, mode='train') -> None:
        super().__init__()
        assert (mode in ['train', 'val', 'test']), f'mode \"{mode}\" is not valid'
        self.mode = mode
        self.files_domainA = list(Path(domainA_path).rglob('{}'.format(name_filter_domainA)))
        self.files_domainB = list(Path(domainB_path).rglob('{}'.format(name_filter_domainB)))
        self.max_data_per_epoch = max_data_per_epoch
        if self.max_data_per_epoch == -1:
            self.max_data_per_epoch = self.dataset_size = max(len(self.files_domainA), len(self.files_domainB))
        else:
            self.dataset_size = min(max(len(self.files_domainA), len(self.files_domainB)), max_data_per_epoch)
        self.small_dataset_start_domainA = 0
        self.small_dataset_start_domainB = 0
        self.invoked_times = 0

        # This transformation function does not only convert the object to a tensor 
        # but also normalizes the data in the range [0.0, 1.0] and changes the shape from (H x W x C) to (C x H x W).
        compose = [transform.ToTensor()]
        if image_size != -1:
            compose.append(MyResize(int(image_size)))
        if crop_size != -1:          
            compose.append(MyRandomCrop(int(crop_size)))
                        
        if self.mode == 'train':
            compose.append(transform.RandomHorizontalFlip())
        # Because "transform.ToTensor" normalizes the input data in the range [0.0, 1.0], the mean and the variance are equal to 0.5.
        compose.append(transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transform.Compose(compose)                        

    def __getitem__(self, index):
        index_A = (self.small_dataset_start_domainA + index) % len(self.files_domainA)
        index_B = (self.small_dataset_start_domainB + index) % len(self.files_domainB)

        # if we are not in the training mode, only use first {self.max_data_per_epoch} data
        if self.invoked_times == self.max_data_per_epoch-1 and self.mode == 'train':
            self.small_dataset_start_domainA = (self.small_dataset_start_domainA + self.max_data_per_epoch) % len(self.files_domainA)
            self.small_dataset_start_domainB = (self.small_dataset_start_domainB + self.max_data_per_epoch) % len(self.files_domainB)
            self.invoked_times = 0
        
        img_domainA = cv2.imread(str(self.files_domainA[index_A]))
        img_domainB = cv2.imread(str(self.files_domainB[index_B]))
        img_domainA = self.transform(img_domainA)
        img_domainB = self.transform(img_domainB)
        self.invoked_times += 1
        return {'A': img_domainA, 'B': img_domainB}

    def __len__(self):
        return self.dataset_size
