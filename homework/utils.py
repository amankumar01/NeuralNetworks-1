from PIL import Image
import torch
import torchvision
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        self.data = list()
        with open(dataset_path + "/labels.csv") as csvfile:
          file_reader = csv.reader(csvfile, delimiter=',')
          index = 0
          for row in file_reader:
            if(index != 0):
              with Image.open(dataset_path + "/" + row[0]) as img:
                tensor_img = transforms.Compose([transforms.ToTensor()])
                self.data.append((tensor_img(img), LABEL_NAMES.index(row[1])))
            index+=1


    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.data[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
