import os
import torch

from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


class ImgsDataset(Dataset):
    """ Cats and dogs dataset """

    def __init__(self, img_names_series, labels_series, img_folder) -> None:
        """
        Args:
            img_names_series (pandas.Series): Image names Series
            labels_series (pandas.Series): Labels Series
            img_folder (string): Directory with images
        """

        self.img_names = img_names_series.to_list()
        self.labels = labels_series.to_list()
        self.dir = img_folder

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.dir,
                                self.img_names[idx])

        image = Image.open(img_name)
        label = self.labels[idx]

        sample = {'image': image, 'label': label}
        return sample
    
