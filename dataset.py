import torch
from torch.utils.data import Dataset
import glob
import cv2
label_dict = {
    "Ballet Flat" : 0,
    "Boat" : 1,
    "Brogue" : 2,
    "Clog" : 3,
    "Sneaker" : 4,
}
class ShoeDataset(Dataset):
    def __init__(self, data_path, img_size=224):
        self.data_path = data_path
        self.img_paths = glob.glob(data_path+'/**/*')
        self.img_size = img_size
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # idx = 1
        img_path = self.img_paths[idx]
        class_type = img_path.split('/')[-2]
        label = label_dict[class_type]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        # print(img)
        img = img / 255
        img = torch.tensor(img).float()
        label = torch.tensor(label).float()
        img = img.permute(2,0,1)
        return img, label

if __name__ == "__main__":
    data_path = "/mnt/nvme0n1p5/Kaggle/archive/Shoes Dataset/Train"
    dataset = ShoeDataset(data_path=data_path)
    print(dataset[11])

