import os
import sys

from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImagenetteDataset(ImageFolder):
    # Store imagenette in memory
    def __init__(self, root, transform=None, target_transform=None):
        img_dict = {}
        super(ImagenetteDataset, self).__init__(root, transform=transform,
            target_transform=target_transform, loader = lambda path: img_dict[path])

        total_size = 0
        for (path, _) in self.imgs:
            img_dict[path] = pil_loader(path)
            total_size += sys.getsizeof(img_dict[path].tobytes())
        print('Imagenette train/val in memory: %0.3fM' % (total_size / 1024.0 / 1024))
        

def get_dataloader(batch_size=128, no_validate=False, dataset='imagenette'):

    data_folder = 'data/' + dataset
    dataset = ImagenetteDataset if dataset == 'imagenette' else ImageFolder
    
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
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=1)

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

    return train_loader, test_loader
