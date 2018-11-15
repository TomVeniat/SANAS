import os
import random
from collections import defaultdict
from enum import Enum

import librosa
import torch
from torch.utils.data import Dataset

from src.commons.utils import timing


class Split(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


class SpeechWordDataset(Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"

    def __init__(self, samples, labels, transform=None, signal_tresh=.5, label_names=None, return_infos=False):
        super(SpeechWordDataset, self).__init__()
        self.samples = samples
        self.labels = labels
        self.transform = transform
        self.signal_tresh = signal_tresh
        self.label_name = label_names
        self.name_label = dict((v, k) for k, v in label_names.items())

        self.label_counts = defaultdict(int)
        for label in labels:
            key = self.label_name[label] if self.label_name else label
            self.label_counts[key] += 1
        self.label_counts = dict(self.label_counts)

        self.return_infos = return_infos

    @classmethod
    def splits(cls, root_dir, wanted_words, unknown_ratio, silence_ratio, silence_duration, splitter):
        word_labels = {word: i + 2 for i, word in enumerate(wanted_words)}
        word_labels.update({cls.LABEL_SILENCE: 0, cls.LABEL_UNKNOWN: 1})

        # sets is a structure in format {"TRAIN": {"Label_A": [x1,x3,x3], ...}, "TEST":...}
        sets = defaultdict(lambda: defaultdict(list))
        bg_noise_files = []
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            is_bg_noise = folder_name == "_background_noise_"
            label = folder_name if folder_name in wanted_words else cls.LABEL_UNKNOWN

            for filename in os.listdir(folder_path):
                wav_name = os.path.join(folder_path, filename)
                if is_bg_noise and os.path.isfile(wav_name) and wav_name.endswith('.wav'):
                    bg_noise_files.append(wav_name)
                else:
                    split = splitter(folder_name, filename)
                    sets[split][word_labels[label]].append(wav_name)

        unkn_label = word_labels[cls.LABEL_UNKNOWN]
        silence_label = word_labels[cls.LABEL_SILENCE]
        for split, data in sets.items():
            n_elem = sum(len(v) for k, v in data.items() if k is not unkn_label)
            n_unkn = int(unknown_ratio * n_elem)

            random.shuffle(data[unkn_label])
            data[unkn_label] = data[unkn_label][:n_unkn]

            n_slience = int(silence_ratio * n_elem)
            data[silence_label] = [torch.zeros(silence_duration)] * n_slience

        lists_ds = cls.split_as_lists(sets)

        return lists_ds[Split.TRAIN], lists_ds[Split.VAL], lists_ds[Split.TEST], bg_noise_files, word_labels

    @staticmethod
    def split_as_lists(splits):
        lists_ds = {}
        for split_name, data in splits.items():
            samples = []
            labels = []
            for y, xs in data.items():
                samples += xs
                labels += [y] * len(xs)
            lists_ds[split_name] = (samples, labels)
        return lists_ds

    @timing
    def __getitem__(self, index):
        sample = self.samples[index]
        if isinstance(sample, str):
            # Sample is the path to the audio file
            sample = torch.from_numpy(librosa.load(sample, sr=None)[0])

        for tr in self.transform:
            sample = tr(sample)

        data, signal_per_conv_frame, sound, mask = sample
        label = self.labels[index]

        per_frame_labels = (signal_per_conv_frame > self.signal_tresh).float() * label

        # todo store sound and mask for exp monitoring

        res = [data, per_frame_labels]

        if self.return_infos:
            res.append(dict(signal_per_frame=signal_per_conv_frame, sound=sound, mask=mask))

        return res

    @property
    def ordered_class_names(self):
        return [v for k, v in sorted(self.label_name.items())]

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        labels = ', '.join(
            ["'{}'({}): {}".format(key, self.name_label[key], count) for key, count in self.label_counts.items()])
        return "SpeechWordFolder dataset containing {} elements. Labelled words are {{{}}}".format(len(self), labels)
