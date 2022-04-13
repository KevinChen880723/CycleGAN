import cv2
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transform

class CycleGANDataset(Dataset):
    def __init__(self, domainA_path, format_domainA, domainB_path, format_domainB, image_size, crop_size) -> None:
        super().__init__()
        self.files_domainA = list(Path(domainA_path).rglob('*.{}'.format(format_domainA)))
        self.files_domainB = list(Path(domainB_path).rglob('*.{}'.format(format_domainB)))
        self.dataset_size = max(len(self.files_domainA), len(self.files_domainB))

        compose = [transform.ToTensor()]
        if image_size != -1:
            compose.append(transform.Resize(int(image_size), transform.InterpolationMode.BICUBIC))
        if crop_size != -1:          
            compose.append(transform.RandomCrop(int(crop_size)))
                        
        compose += [transform.RandomHorizontalFlip(),
                    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transform.Compose(compose)                        

    def __getitem__(self, index):
        index_A = index % len(self.files_domainA)
        index_B = index % len(self.files_domainB)
        img_domainA = cv2.imread(str(self.files_domainA[index_A]))
        img_domainB = cv2.imread(str(self.files_domainB[index_B]))
        img_domainA = self.transform(img_domainA)
        img_domainB = self.transform(img_domainB)
        return {'A': img_domainA, 'B': img_domainB}

    def __len__(self):
        return self.dataset_size
