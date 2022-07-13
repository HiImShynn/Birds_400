import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import io

import os
import pandas as pd

def get_data_partition(csv_file, partition= None):
    if partition:
        raw_data = pd.read_csv(csv_file)
    return raw_data[raw_data.data_set == partition]

class Birds(Dataset):
    def __init__(self, images_dir, annotations_file, data_partition= None, transform= None):
        self.images_dir = images_dir
        self.annotations = get_data_partition(annotations_file, data_partition)
        self.transform = transform
        self.data_partition = data_partition

    def __getitem__(self, index):
        image_dir = os.path.join(self.images_dir, self.annotations.iloc[index, 1])
        image = io.imread(image_dir)
        label = torch.tensor(int(self.annotations.iloc[index, 0]))

        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.annotations)

if __name__ == "__main__":
    trans = transforms.Compose([transforms.ToTensor()])
    data = Birds("", "birds.csv", "test", transform= trans)
    data_train = DataLoader(data, batch_size= 16)
    image, label = next(iter(data_train))
    # print(os.getcwd() + "/birds.csv")
    print(image)
    print(image.shape)
    