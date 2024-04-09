import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

class CamvidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.mask_transform=mask_transform
        self.images=os.listdir(image_dir)
        self.masks=os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform:
            image=self.transform(image)
        if self.mask_transform:
            mask=self.mask_transform(mask)

        return {'input': image, 'label': mask}
    
if __name__=='__main__':
    train_path = 'Camseq01/train'
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    
    dataset = CamvidDataset(image_dir=os.path.join(train_path, 'image'),
                            mask_dir=os.path.join(train_path, 'mask'),
                            transform=transform, mask_transform=mask_transform)
    
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(2):
        print(f"epoch : {epoch} ")
        for batch in data_loader:
            img, label = batch
            print(img.size(), label)
    