from scripts.pytorch_ds import ModelNet10PC
from torch.utils.data import DataLoader

dataset = ModelNet10PC("data/modelnet10_pc_2048")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    print(batch.shape)
    break
