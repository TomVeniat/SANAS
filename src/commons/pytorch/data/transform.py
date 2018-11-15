import random

import librosa
import numpy as np
import torch
import torch.nn.functional as F


class Pad1d(object):
    def __init__(self, total_length):
        super(Pad1d, self).__init__()
        self.total_length = total_length

    def __call__(self, data):
        assert data.dim() == 1
        input_len = data.size(0)
        pad_size = self.total_length - input_len
        assert pad_size >= 0

        return F.pad(data, (0, pad_size)).data


class RandomNoise(object):
    def __init__(self, noise_files, sample_time_min, sample_time_max, noise_proba, nsr_max, sr):
        """

        :param noise_files: List of absolute paths from which the background sounds will be selected.
        :param sample_time_min: The minimal final length is seconds of the processed sample.
        :param sample_time_max: The maximal final length is seconds of the processed sample.
        :param noise_proba: The probability of adding a background noise to the sound.
        :param nsr_max: The noise to signal maximal ratio.
        :param sr: The sample rate used.
        """
        super(RandomNoise, self).__init__()
        self.noise_files = noise_files
        self.min_samples = sample_time_min * sr
        self.max_samples = sample_time_max * sr
        self.noise_proba = noise_proba
        self.nsr_max = nsr_max
        self.sample_rate = sr

        self.bg_noise_audio = [torch.from_numpy(librosa.load(file, sr=None)[0]) for file in self.noise_files]


    def __call__(self, sound):
        sample_len = random.randint(self.min_samples, self.max_samples)

        sample = torch.zeros(sample_len)
        contains_sound_mask = torch.zeros(sample_len)  # .byte()

        if random.random() < self.noise_proba:
            bg_noise = random.choice(self.bg_noise_audio)

            noise_start = random.randint(0, bg_noise.size(0) - sample_len)
            bg_noise = bg_noise[noise_start:noise_start + sample_len]

            nsr = random.random() * self.nsr_max
            sample += bg_noise * nsr

        signal_start = random.randint(0, sample_len - sound.size(0))

        sample[signal_start:signal_start + sound.size(0)] += sound
        contains_sound_mask[signal_start:signal_start + sound.size(0)] = 1

        return torch.clamp(sample, -1, 1), contains_sound_mask


class TimeShift(object):
    def __init__(self, max_shift):
        super(TimeShift, self).__init__()
        self.max_shift = max_shift

    def __call__(self, data):
        shift = random.randint(-self.max_shift, self.max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = F.pad(data, (a, b)).data
        return data[:shift] if shift < 0 else data[shift:]


class MFCC(object):
    def __init__(self, n_mel_filters, n_dct_filters, sr, n_fft, hop_length, return_sound=False, use_labels=False,
                 **kwargs):
        """

        :param n_mel_filters:
        :param n_dct_filters:
        :param frame_lengt:
        :param frame_stride:
        :param sr:
        """
        super(MFCC, self).__init__()
        self.n_mel_filters = n_mel_filters
        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mel_filters)
        self.sr = sr

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.f_min = 20
        self.f_max = 4000

        self.return_sound = return_sound
        self.use_labels = use_labels

    def __call__(self, data):
        if self.use_labels:
            sound, mask, labels = data
        else:
            sound, mask = data

        features = librosa.feature.melspectrogram(sound.numpy(), sr=self.sr, n_mels=self.n_mel_filters,
                                                  hop_length=self.hop_length, n_fft=self.n_fft,
                                                  fmin=self.f_min, fmax=self.f_max)
        features[features > 0] = np.log(features[features > 0])
        features = [np.matmul(self.dct_filters, x) for x in np.split(features, features.shape[1], axis=1)]
        features = np.expand_dims(np.array(features, order="F").squeeze(2), 0).astype(np.float64)
        features = torch.as_tensor(features, dtype=torch.float32)

        # Pad the mask to have the same length as the signal taken by librosa stft (frames are centered and not
        # left-alignend in stft calls)
        padded_mask = F.pad(mask, (self.n_fft // 2,) * 2)

        n_frames = (len(padded_mask) - self.n_fft) // self.hop_length + 1
        assert features.size(1) == n_frames

        signal_ratio_per_stft_frame = torch.stack(
            [padded_mask[i * self.hop_length: i * self.hop_length + self.n_fft].mean() for i in range(n_frames)])

        if self.use_labels:
            padded_labels = F.pad(labels, (self.n_fft // 2,) * 2)
            label_per_stft_frame = torch.stack(
                [padded_labels[i * self.hop_length: i * self.hop_length + self.n_fft].max() for i in range(n_frames)])
            return features, signal_ratio_per_stft_frame, label_per_stft_frame, sound, mask

        else:
            return features, signal_ratio_per_stft_frame, sound, mask


class SequenceTransform(object):
    def __init__(self, frame_len=100, stride=20):
        self.stride = stride
        self.frame_len = frame_len

    def __call__(self, data):
        features, signal_ratio_per_stft_frame, sound, mask = data
        n_frames = (features.size(1) - self.frame_len) // self.stride + 1
        res = torch.stack([features[:, i * self.stride:i * self.stride + self.frame_len, :] for i in range(n_frames)])
        signal_ratio_per_conv_frame = torch.stack(
            [signal_ratio_per_stft_frame[i * self.stride:i * self.stride + self.frame_len].mean() for i in
             range(n_frames)])
        return res, signal_ratio_per_conv_frame, sound, mask
