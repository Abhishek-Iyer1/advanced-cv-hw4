import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio

class NIST36(Dataset):
    def __init__(self, filepath:str, type:str):
        data = sio.loadmat(filepath)
        self.x = data[f"{type}_data"] # 1024
        self.one_hot_y = data[f"{type}_labels"] # 36
        self.y = np.argmax(self.one_hot_y, axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = torch.from_numpy(self.x[index]).type(torch.float32)
        label = torch.tensor(self.y[index]).type(torch.LongTensor)
        return image, label