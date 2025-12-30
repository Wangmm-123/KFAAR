import os
import os.path as osp
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple


class LFW(Dataset):
    def __init__(self, root='E:/new', mode='train'):
        super().__init__()
        self.root = osp.expanduser(root)
        if mode == 'train':
            data = 'LFW/train.txt'
        elif mode == 'dev':
            data = 'LFW/dev.txt'
        elif mode == 'test':
            data = 'LFW/test.txt'
        else:
            raise Exception('Mode Error!')

        with open(data, 'r') as f:
            self.triplets = f.readlines()
        
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __path2tensor__(self, path: str) -> torch.Tensor:
        path = osp.join(self.root, path)
        img = Image.open(path).convert("RGB")
        # w, h = img.size
        # s = min(w, h)
        # img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        # img = img.resize((256, 256), Image.LANCZOS)
        img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        triplet = self.triplets[index]
        pth_x1, pth_x2, pth_y = triplet.split()
        x1 = self.__path2tensor__(pth_x1)
        x2 = self.__path2tensor__(pth_x2)
        y = self.__path2tensor__(pth_y)

        return x1, x2, y


class CelebA(Dataset):
    def __init__(self, root='F:/CelebA/Img/img_align_celeba_128', mode='train'):
        super().__init__()
        self.root = osp.expanduser(root)
        if mode == 'train':
            data = 'CelebA/n_train.txt'
        elif mode == 'dev':
            data = 'CelebA/n_dev.txt'
        elif mode == 'test':
            data = 'CelebA/n_test.txt'
        else:
            raise Exception('Mode Error!')
        
        with open(data, 'r') as f:
            self.triplets = f.readlines()
        
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __path2tensor__(self, path: str) -> torch.Tensor:
        path = osp.join(self.root, path)
        img = Image.open(path).convert('RGB')
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((256, 256), Image.LANCZOS)
        img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        triplet = self.triplets[index]
        pth_x1, pth_x2, pth_y = triplet.split()
        x1 = self.__path2tensor__(pth_x1)
        x2 = self.__path2tensor__(pth_x2)
        y = self.__path2tensor__(pth_y)
        return x1, x2, y


class VirtualDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = osp.expanduser(root)
        ids = os.listdir(self.root)
        assert len(ids) % 2 == 0

        self.triplets = []
        for i in range(int(len(ids)/2)):
            x_dir = osp.join(self.root, f'{2*i}')
            y_dir = osp.join(self.root, f'{2*i+1}')
            x = os.listdir(x_dir)
            y = os.listdir(y_dir)
            self.triplets.append([osp.join(x_dir, x[0]), osp.join(x_dir, x[1]), osp.join(y_dir, y[0])])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __path2tensor__(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((256, 256), Image.LANCZOS)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        x1, x2, y = self.triplets[index]
        x1 = self.__path2tensor__(x1)
        x2 = self.__path2tensor__(x2)
        y = self.__path2tensor__(y)
        return x1, x2, y


class LFWTestMono(Dataset):
    def __init__(self, root='E:/lfw') -> None:
        super().__init__()
        self.root = osp.expanduser(root)

        with open('LFW/test.txt') as f:
            lines = f.readlines()
        imgs = set()
        for line in lines:
            triplet = line.strip().split()
            for img in triplet:
                imgs.update([img])
        self.imgs = list(imgs)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        img = Image.open(osp.join(self.root, self.imgs[index])).convert('RGB')
        img = self.transform(img)
        return img


class CelebATestMono(Dataset):
    def __init__(self, root='dataset/CelebA') -> None:
        super().__init__()
        self.root = osp.expanduser(root)

        with open('CelebA/test.txt') as f:
            lines = f.readlines()

        imgs = set()
        for line in lines:
            triplet = line.strip().split()
            for img in triplet:
                imgs.update([img])
        self.imgs = list(imgs)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        img = Image.open(osp.join(self.root, self.imgs[index])).convert('RGB')
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((256, 256), Image.LANCZOS)
        img = self.transform(img)
        return img



    # def __len__(self):
    #     return len(self.imgs)
    #
    # def __getitem__(self, index: int) -> List[torch.Tensor]:
    #     img = Image.open(osp.join(self.root, self.imgs[index])).convert('RGB')
    #     w, h = img.size
    #     s = min(w, h)
    #     img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    #     img = img.resize((256, 256), PIL.Image.LANCZOS)
    #     img = self.transform(img)

        return img


if __name__ == "__main__":
    celeba = CelebA(mode='dev')
    x1, x2, y = celeba[0]
    print(x1.shape, x2.shape, y.shape)
    print(x1.device)

    lfw = LFW(mode='dev')
    x1, x2, y = lfw[0]
    print(x1.shape, x2.shape)

    vir = VirtualDataset(root='virtual_train')
    x1, x2, y = vir[0]
    print(x1.shape, x2.shape, y.shape)

    mono = LFWTestMono(root='E:/lfw')
    x = mono[0]
    print(x.shape)
