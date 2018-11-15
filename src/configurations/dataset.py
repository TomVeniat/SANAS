import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.commons.pytorch.data.SpeechWordDataset import SpeechWordDataset
from src.commons.pytorch.data.Splitter import SpeechCommandsSplitter
from src.commons.pytorch.data.collate import PadCollate
from src.commons.pytorch.data.transform import RandomNoise, MFCC, SequenceTransform

logger = logging.getLogger(__name__)


def compute_class_weight(data_loader, n_classes):
    counts = torch.zeros(n_classes)
    for sample in tqdm(data_loader, desc='Counting'):
        y = sample[1]
        for i in range(n_classes):
            counts[i] += (y == i).sum().item()

    tot = counts.sum()
    logger.info(counts)
    weights = tot / (counts * n_classes)
    return weights


def data_config():
    num_workers = 8
    batch_size = 64

    root_path = './data/speech_commands_v0.01'
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    n_classes = 2 + len(words)

    unknown_ratio = .1
    silence_ratio = .1

    noise_proba_train = .8
    noise_proba_test = .8
    nsr_max = .3

    n_mel_filters = 40
    n_dct_filters = 40

    clean_eval = False
    batch_first = False

    frame_len = 101
    input_dim = (1, frame_len, n_dct_filters)

    shuffle = True


def speech_commands():
    dataset = 'speech_commands'

    sr = 16000
    signal_treshold = .5

    clean_eval = False

    n_fft = 480
    hop_length = 160

    seq_len_min = 1
    seq_len_max = 3

    n_mel_filters = 40
    n_dct_filters = 40

    frame_stride = 20

    balance_classes = True


def create_speech_commands(root_path, words, unknown_ratio, silence_ratio, n_fft, hop_length, seq_len_min,
                           seq_len_max, frame_len, frame_stride, noise_proba_train, noise_proba_test, nsr_max,
                           signal_treshold, sr, batch_size, n_mel_filters, n_dct_filters, num_workers, n_classes,
                           balance_classes, clean_eval, shuffle, **kwargs):
    batch_first = False

    logger.info('Looking for data at {}'.format(root_path))

    splitter = SpeechCommandsSplitter(10, 10)
    train_d, val_d, test_d, noise_d, labels = SpeechWordDataset.splits(root_path, words, unknown_ratio,
                                                                       silence_ratio, sr, splitter)
    labels = {v: k for k, v in labels.items()}

    train_transformations = [
        # Pad1d(sr),
        # TimeShift(sr * time_shift_ms / 1000),
        RandomNoise(noise_d, seq_len_min, seq_len_max, noise_proba_train, nsr_max, sr),
        MFCC(n_mel_filters, n_dct_filters, sr, n_fft, hop_length, return_sound=True),
        SequenceTransform(frame_len=frame_len, stride=frame_stride)
    ]

    eval_transformations = [
        RandomNoise(noise_d, seq_len_min, seq_len_max, 0 if clean_eval else noise_proba_test, nsr_max, sr),
        MFCC(n_mel_filters, n_dct_filters, sr, n_fft, hop_length, return_sound=True),
        SequenceTransform(frame_len=frame_len, stride=frame_stride)
    ]

    train_ds = SpeechWordDataset(*train_d, train_transformations, signal_treshold, labels, return_infos=True)
    val_ds = SpeechWordDataset(*val_d, eval_transformations, signal_treshold, labels, return_infos=True)
    test_ds = SpeechWordDataset(*test_d, eval_transformations, signal_treshold, labels, return_infos=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,

        collate_fn=PadCollate(batch_first=batch_first), num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=PadCollate(batch_first=batch_first), num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle,
                             collate_fn=PadCollate(batch_first=batch_first), num_workers=num_workers)

    if balance_classes and seq_len_max > 1:
        # dataset is imbalanced (More silences than other labels
        class_weights = compute_class_weight(train_loader, n_classes)
    else:
        class_weights = torch.ones(n_classes)

    logger.info('Class weights: {}'.format(class_weights))

    datasets = dict(train=train_loader, validation=val_loader, test=test_loader)
    for k, v in datasets.items():
        logger.info('{:11}: {}'.format(k.title(), v.dataset))

    return datasets, batch_first, class_weights


ds_funcs = {
    'speech_commands': create_speech_commands
}


def select_dataset(dataset, **kwargs):
    return ds_funcs[dataset](**kwargs)


def dataset_config(ex):
    ex.config(data_config)

    ex.named_config(speech_commands)

    ds_funcs['speech_commands'] = ex.capture(create_speech_commands)

    select_dataset_capt = ex.capture(select_dataset)

    return select_dataset_capt
