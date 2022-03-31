import os
import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImgsDataset(Dataset):
    """ Cats and dogs dataset """

    def __init__(self, img_names_series, labels_series, img_folder, transforms_ = None) -> None:
        """
        Args:
            img_names_series (pandas.Series): Image names Series
            labels_series (pandas.Series): Labels Series
            img_folder (string): Directory with images
        """

        self.img_names = img_names_series.to_list()
        self.labels = labels_series.to_list()
        self.dir = img_folder
        self.transforms = transforms_


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.dir,
                                self.img_names[idx])

        image = Image.open(img_name)

        if self.transforms:
            image = self.transforms(image)
            image = image.numpy()
            return image.astype('float32'), self.labels[idx]
        else:
            self.transforms_ = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
            image = self.transforms_(image)
            image = image.numpy()
            return image.astype('float32'), self.labels[idx]



        
        #return sample
        return trsfmd_img, label
    
