from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random

class ModelNet10PC(Dataset):
    def __init__(self, root, split="train", seed=42):
        """
        root: directory with class folders containing .npy point clouds.
        split: 'train', 'val', or 'test'
        """
        random.seed(seed)
        self.split = split

        self.train_files = []
        self.val_files   = []
        self.test_files  = []

        classes = sorted(os.listdir(root))

        for cls in classes:
            class_dir = os.path.join(root, cls)
            if not os.path.isdir(class_dir):
                continue

            # gather ALL .npy for this class
            files = glob.glob(os.path.join(class_dir, "*.npy"))
            if len(files) == 0:
                # Also check subfolders (in case preprocessing kept train/test/)
                files = glob.glob(os.path.join(class_dir, "**/*.npy"), recursive=True)

            files = sorted(files)
            random.shuffle(files)

            n = len(files)
            n_train = int(0.70 * n)
            n_val   = int(0.10 * n)
            n_test  = n - n_train - n_val   # ≈20%

            self.train_files += files[:n_train]
            self.val_files   += files[n_train:n_train + n_val]
            self.test_files  += files[n_train + n_val:]

        if split == "train":
            self.files = self.train_files
        elif split == "val":
            self.files = self.val_files
        elif split == "test":
            self.files = self.test_files
        else:
            raise ValueError("split must be: train / val / test")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc = np.load(self.files[idx]).astype(np.float32)

        # ---- Normalize to unit sphere ----
        pc = pc - pc.mean(axis=0)
        pc = pc / np.max(np.linalg.norm(pc, axis=1))

        return pc



