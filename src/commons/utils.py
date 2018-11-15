import logging
import os
import shutil
import tempfile
import time

import torch

logger = logging.getLogger(__name__)


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.debug('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def print_properties(viz, props):
    properties = [{'type': 'text', 'name': k, 'value': str(v)} for k, v in sorted(props.items())]
    viz.properties(properties)


def compute_entropy(probas):
    zero_entropy_mask = (probas == 0) + (probas == 1)
    neg_probas = torch.ones_like(probas) - probas
    entropies = - probas * probas.log2() - neg_probas * neg_probas.log2()
    entropies[zero_entropy_mask] = 0
    return entropies


def compute_discounted_rewards(rewards, gamma):
    """
    :param rewards: seq_len*batch_size
    :param gamma: the discount factor
    :return: a seq_len*batch_size
    """
    seq_len = rewards.size(0)
    advantage = rewards.clone().detach()
    for i in range(seq_len - 2, -1, -1):
        advantage[i, :] += advantage[i + 1, :] * gamma
    return advantage


class EMA(object):
    def __init__(self, beta):
        super(EMA, self).__init__()
        self.beta = beta
        self.val = None

    def update(self, value):
        if self.val is None:
            self.val = value
        else:
            self.val = self.beta * value + (1 - self.beta) * self.val
        return self.val

    def reset(self):
        self.val = None


def save_checkpoint(model, ex, epoch):
    save_folder = tempfile.mkdtemp()
    file_name = 'exp_{}_epoch_{}.chkpt'.format(ex.current_run._id, epoch)
    save_path = os.path.join(save_folder, file_name)

    logger.info('Saving checkpoint: {}'.format(save_path))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # To avoid GPU memory pollution
    state = dict((k, v.cpu()) for k, v in model.state_dict().items())
    torch.save(state, save_path)
    ex.add_artifact(save_path, file_name)
    shutil.rmtree(save_folder)

    return file_name


@timing
def print_batch(xs, ys, sounds, masks, label_name_map, batch_first, viz):
    """
    Bunch of ugly things plotting the input images and sound signal + sending the sound audio to visdom.

    """
    split_dim = 0 if batch_first else 1
    seq_xs = xs.split(1, dim=split_dim)
    seq_ys = ys.split(1, dim=split_dim)
    for i, (x, y, sound, mask) in enumerate(zip(seq_xs, seq_ys, sounds, masks)):
        x, y = (x.squeeze(split_dim), y.squeeze(split_dim))

        label_n = '{}: {}'.format(i, label_name_map[y.max().item()])

        x_red = x.clone()
        x_red[y > 0] = -255

        x_green = x.clone()
        x_green[y == 0] = -255

        x_blue = torch.ones(x.size()) * -255

        x = torch.cat([x_red, x_green, x_blue], dim=1)

        images = x.transpose(-1, -2)
        viz.images(images, opts={'title': label_n}, nrow=11, padding=3)

        sr = 16000
        viz.audio(sound, opts={'sample_frequency': sr, 'title': label_n, 'height': 350})
        x = torch.arange(0, sound.size(0)).float() / sr

        line_pos = torch.empty(sound.size(0)).fill_(float('nan'))
        line_neg = torch.empty(sound.size(0)).fill_(float('nan'))
        line_pos[mask == 1] = sound[mask == 1]
        line_neg[mask == 0] = sound[mask == 0]

        viz.line(torch.stack([line_neg, line_pos], dim=1), X=x, name='amp',
                 opts=dict(
                     title=label_n,
                     xlabel='Time (s)', xtickfont={'size': 19},
                     legend=['Background noise', 'Word + Background noise'],
                     layoutopts={'plotly': {
                         'autosize': True,
                         'yaxis': {
                             'automargin': True,
                             'title': 'Signal Amplitude',
                             'tickfont': {'size': 15}},
                         'font': {
                             'family': 'Times New Roman',
                             'size': 20},
                         'legend': {
                             'x': 0.7,
                             'y': 0.1,
                             'font': {'size': 15}}}},
                     traceopts={'plotly': {
                         'Background noise': {
                             'line': {
                                 'color': 'rgb(200,30,30)'}},
                         'Word + Background noise': {
                             'line': {
                                 'color': 'rgb(30,200,30)'}}}}))
