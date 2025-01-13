import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, x, y):
        
        self.x, self.y = x, y

    def __len__(self):
        
        return len(self.x)

    def __getitem__(self, idx):
        
        time = torch.tensor(self.x[idx, 0]).type(torch.int) 
        x = torch.tensor(self.x[idx, 1:]).type(torch.float32)
        y = torch.tensor(self.y[idx]).type(torch.int)
        
        return time, x, y