import torch.utils.data as tud
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(tud.Dataset):
    def __init__(self, imgs, Apow, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.Apow = Apow
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # dsm_f = self.dsm[index]
        Apow_n = self.Apow[index,:,:] #3D ndarray
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label,Apow_n

    def __len__(self):
        return len(self.imgs)