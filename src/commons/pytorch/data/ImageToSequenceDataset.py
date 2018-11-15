import numpy as np
from torch.utils import data


class SeqDataset(data.Dataset):
    def __init__(self, source_ds, min_seq_len, max_seq_len, stride, frame_size, pad_value=0, transform=None,
                 target_transform=None):
        super(SeqDataset, self).__init__()
        self.source_ds = source_ds
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.frame_size = frame_size
        self.pad_value = pad_value
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        source_x, source_y = self.source_ds[index]
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        size = self.stride * (seq_len - 1) + self.frame_size
        start_index = np.random.randint(0, len(self.pad_value) - seq_len)
        background = self.pad_value[start_index, start_index + self.frame_size]

    def __len__(self):
        return len(self.source_ds)
