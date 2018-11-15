import hashlib
import os
import re

from src.commons.pytorch.data.SpeechWordDataset import Split


class Splitter(object):
    def __call__(self, label, sample):
        """
        Returns the Split for the given Sample
        """
        raise NotImplementedError


class ListSplitter(Splitter):
    def __init__(self, root, val_path, test_path):
        super(ListSplitter, self).__init__()

        with open(os.path.join(root, val_path), 'r') as f:
            self.val_samples = set(line.rstrip() for line in f)

        with open(os.path.join(root, test_path), 'r') as f:
            self.test_samples = set(line.rstrip() for line in f)

    def __call__(self, label, sample):
        item = os.path.join(label, sample)
        if item in self.val_samples:
            split = Split.VAL
        elif item in self.test_samples:
            split = Split.TEST
        else:
            split = Split.TRAIN
        return split


class SpeechCommandsSplitter(Splitter):
    def __init__(self, val_percentage, test_percentage):
        super(SpeechCommandsSplitter, self).__init__()
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage

    def __call__(self, label, sample):
        hashname = re.sub(r"_nohash_.*$", "", sample)
        max_no_wavs = 2 ** 27 - 1
        bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
        bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
        if bucket < self.val_percentage:
            split = Split.VAL
        elif bucket < self.test_percentage + self.val_percentage:
            split = Split.TEST
        else:
            split = Split.TRAIN
        return split
