import csv
from collections import Counter, defaultdict

import librosa
import torch
from torch.utils.data import Dataset

from src.commons.pytorch.data.transform import MFCC


class StreamWordDataset(Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"

    def __init__(self, sample, label_file, transform=None, signal_tresh=.5, label_names=None, frame_len=480, frame_stride=160,
                 return_infos=False, single_seq=False, **kwargs):
        super(StreamWordDataset, self).__init__()
        self.sample, self.sr = librosa.load(sample, sr=None)
        self.sample = torch.from_numpy(self.sample)
        self.label_mask = torch.zeros_like(self.sample)
        self.labels = torch.zeros_like(self.sample).long()
        self.label_names = label_names
        self.label_names_rev = {txt: i for i, txt in enumerate(self.label_names)}
        self.timed_labels=[]

        with open(label_file, encoding="utf-8") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for word, time in spamreader:
                label = self.label_names_rev[word]
                start_time = int(float(time) * self.sr / 1000)
                end_time = int(start_time + self.sr)
                self.labels[start_time:end_time] = label
                self.label_mask[start_time:end_time] = 1
                self.timed_labels.append((word, float(time)))

        self.transform = transform
        self.signal_tresh = signal_tresh
        self.single_seq = single_seq

        data = (self.sample, self.label_mask, self.labels)
        for t in self.transform:
            data = t(data)
        features, signal_ratio_per_stft_frame, label_per_stft_frame, sound, mask = data
        self.features = features
        self.signal_ratio_per_stft_frame = signal_ratio_per_stft_frame
        self.label_per_stft_frame = label_per_stft_frame

        self.final_labels = label_per_stft_frame * (signal_ratio_per_stft_frame>self.signal_tresh).long()

        self.frame_len = frame_len
        self.frame_stride = frame_stride

        self.count = None

        #self.label_counts = defaultdict(int)
        # for label in labels:
        #     key = self.label_name[label] if self.label_name else label
        #     self.label_counts[key] += 1
        # self.label_counts = dict(self.label_counts)
        #
        # self.return_infos = return_infos

    def _extract_step(self, index):
        start = index * self.frame_stride
        end = start + self.frame_len

        has_signal = self.signal_ratio_per_stft_frame[start:end].mean() > self.signal_tresh
        if has_signal:
            c = Counter(e.item() for e in self.label_per_stft_frame[start:end])
            if len(c) <= 2 or c.most_common(1)[0][1] > self.frame_len/2:
                label = torch.tensor([(c.most_common(1)[0][0])])
            else:
                label = torch.ones(1).long() * self.label_names_rev['_silence_']
        else:
            label = torch.ones(1).long() * self.label_names_rev['_silence_']

        return self.features[:, start:end], label

    def __getitem__(self, index):
        if self.single_seq:
            assert index == 0
            return self._whole_seq()
        else:
            return self._extract_step(index)

    def _n_steps(self):
        # return 150
        return (self.features.size(1) - self.frame_len) // self.frame_stride + 1

    def __len__(self):
        if self.single_seq:
            return 1
        else:
            return self._n_steps()

    def _init_count(self):
        self.count = defaultdict(int)
        for i in range(self._n_steps()):
            _, y = self._extract_step(i)
            self.count[self.label_names[y.item()]] += 1

        tot = sum(self.count.values())
        self.norm_count = {k: v / tot for k, v in self.count.items()}


    def get_count(self, normalize=False):
        if self.count is None:
            self._init_count()

        if normalize:
            return self.norm_count
        else:
            return self.count

    def _whole_seq(self):
        xs = []
        ys = []
        for t in range(self._n_steps()):
            x_t, y_t = self._extract_step(t)
            xs.append(x_t)
            ys.append(y_t)
        xs = torch.stack(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        return xs, ys, []



if __name__ == '__main__':
    SAMPLE_PATH = '/local/veniat/data/speech/speech_commands_streaming_test_v0.02/'
    SOUND = 'streaming_test.wav'
    LABELS = 'streaming_test_labels.txt'

    label_names = ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    n_mel_filters = 40
    n_dct_filters = 40
    sr = 16000
    n_fft = 480
    hop_length = 160

    signal_tresh=.5

    frame_len = 101
    frame_stride = 20

    transforms = [MFCC(n_mel_filters, n_dct_filters, sr, n_fft, hop_length, return_sound=True, use_labels=True)]

    ds = StreamWordDataset(SAMPLE_PATH + SOUND, SAMPLE_PATH + LABELS, transform=transforms,
                           signal_tresh=signal_tresh, label_names=label_names, frame_len=frame_len, frame_stride=frame_stride)
    print(ds[0])
    print(ds[0][0].shape)
    print(ds)

    for i in range(len(ds)):
        x, y = ds[i]
        print(y)

    print(ds.get_count())
