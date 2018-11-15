"""from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/"""
import torch


def pad_tensor(vec, pad, dim, value=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec.float(), torch.ones(*pad_size) * value], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0, batch_first=False, ignore_value=-7):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.batch_first = batch_first
        self.ignore_value = ignore_value

    def pad_collate(self, batch):
        """
        args:
            batch - list of (sample, label) elements. sample size: (seq_len, feature_dims...). label size: (seq_len)

        return:
            xs - a tensor containing all examples in 'batch' after padding, size: (batch_size, seq_len, feature_dims...)
             if batch_first else (seq_len, batch_size, feature_dims...)
            ys - a LongTensor of all labels in batch, size: (batch_size, seq_len)
        """
        # find longest sequence
        lengths = map(lambda sample: sample[0].size(self.dim), batch)
        max_len = max(lengths)

        # pad each sample to max_len
        xs, ys, infos = zip(*map(
            lambda sample: (pad_tensor(sample[0], pad=max_len, dim=self.dim, value=0),
                            pad_tensor(sample[1], pad=max_len, dim=self.dim, value=self.ignore_value),
                            sample[2:]),
            batch))

        stack_dim = 0 if self.batch_first else 1
        xs = torch.stack(xs, dim=stack_dim)
        ys = torch.stack(ys, dim=stack_dim).long()

        return xs, ys, infos

    def __call__(self, batch):
        return self.pad_collate(batch)
