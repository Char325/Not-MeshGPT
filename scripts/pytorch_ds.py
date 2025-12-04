from torch.utils.data import Dataset
import numpy as np
import glob

class ModelNet10PC(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(root + "/**/*.npy", recursive=True))

    def __len__(self):
        return len(self.files)
    """
    def __getitem__(self, idx):
        pc = np.load(self.files[idx]).astype(np.float32)

        # normalize to unit sphere
        pc = pc - np.mean(pc, axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(pc, axis=1))
        pc = pc / scale

        return pc
    """

    def __getitem__(self, idx):
        pc = np.load(self.files[idx]).astype(np.float32)

        # --- Normalize ---
        # center
        pc = pc - np.mean(pc, axis=0, keepdims=True)

        # scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / max_dist

        return pc


