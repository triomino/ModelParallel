import os
import sys
import random

import numpy

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def random_image(height=8, width=8):
    rgb_array = numpy.random.rand(3, height, width) * 255
    return rgb_array.astype('float32')

class ImagenetteDataset(ImageFolder):
    # Store imagenette in memory
    def __init__(self, root, transform=None, target_transform=None):
        self.img_dict = []
        super(ImagenetteDataset, self).__init__(root, transform=transform,
            target_transform=target_transform)

        total_size = 0
        for (path, _) in self.imgs:
            image = pil_loader(path)
            target = self.class_to_idx[path.split('/')[3]]
            self.img_dict.append((image, target))
            total_size += sys.getsizeof(image.tobytes())
        print('Imagenette train/val in memory: %0.3fM' % (total_size / 1024.0 / 1024))
    
    def __getitem__(self, index):
        image, target = self.img_dict[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
    
    def __len__(self):
        return len(self.imgs)

class RandomDataset(ImageFolder):
    def __init__(self, root, *args, **kwargs):
        self.img_dict = []
        super(RandomDataset, self).__init__(root)

        total_size = 0
        for (path, _) in self.imgs:
            image = random_image()
            self.img_dict.append((image, random.randint(0, 9)))
            total_size += sys.getsizeof(image.tobytes())
        print('Random dataset in memory: %0.3fM' % (total_size / 1024.0 / 1024))

        

    def __getitem__(self, index):
        return self.img_dict[index]
    
    def __len__(self):
        return len(self.imgs)

def get_dataloader(batch_size=128, no_validate=False, dataset='imagenette', distributed=False):

    data_folder = 'data/' + (dataset if dataset != 'random' else 'imagenette')

    dataset = {'imagenette': ImagenetteDataset, 
        'imagenet': ImageFolder,
        'random': RandomDataset
    }[dataset]
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(data_folder, 'train')
    train_set = dataset(train_folder, transform = train_transform)
    if distributed:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              pin_memory=True,
                              num_workers=1,
                              sampler=train_sampler)

    test_loader = None
    if not no_validate:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        test_folder = os.path.join(data_folder, 'val')
        test_set = dataset(test_folder, transform = test_transform)
        test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True)

    if distributed:
        return train_loader, test_loader, train_sampler
    else:
        return train_loader, test_loader
