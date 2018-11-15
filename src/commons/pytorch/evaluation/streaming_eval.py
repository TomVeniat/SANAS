import torch
import logging

from torchnet import meter

from src.commons import external_resources
from src.commons.pytorch.data.streaming_dataset import StreamWordDataset
from src.commons.pytorch.data.transform import MFCC
from src.commons.pytorch.evaluation.StreamingAccuracy import StreamingAccuracy
from src.commons.pytorch.evaluation.evaluation import evaluate_model
from src.commons.sn.implementation.ComputationCostEvaluator import ComputationCostEvaluator
from src.commons.sn.interface.PathRecorder import PathRecorder
from src.configurations.model import create_model

logger = logging.getLogger(__name__)


def eval_model_on_file(id, epoch, data_path, audio_file, label_file, device, use_acc_stats, len_ms, mongo_path,
                       return_pred_stats=False):
    mongo_collection = external_resources.get_mongo_collection(mongo_path=mongo_path)
    res = mongo_collection.find_one({'_id': id})

    config = res['config']

    label_names = ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    stream_transform = [MFCC(return_sound=True, use_labels=True, **config)]

    ds = StreamWordDataset(data_path + audio_file, data_path + label_file, transform=stream_transform,
                           label_names=label_names, single_seq=True, **config)

    batch_first = False
    logger.info(ds.get_count())
    logger.info(ds.get_count(True))

    xs, ys, other = ds[0]
    xs = xs.unsqueeze(1)
    ys = ys.unsqueeze(1)

    if len_ms > 0:
        xs = xs[:len_ms // (config['frame_stride'] * 10)]
        ys = ys[:len_ms // (config['frame_stride'] * 10)]

    dl = [(xs, ys, other)]

    logger.info('Dataset ok')

    ####MODEL####
    start_chkpt = 'exp_{}_epoch_{}.chkpt'.format(id, epoch)
    config['start_chkpt'] = start_chkpt
    config['sep_features'] = config.get('sep_features', False)

    model = create_model(**config, mongo_config_path=mongo_path)

    path_recorder = PathRecorder(model.stochastic_model)
    cost_evaluator = ComputationCostEvaluator(node_index=path_recorder.node_index, bw=False)

    cost_evaluator.init_costs(model.stochastic_model)
    logger.info('Cost: {:.5E}'.format(cost_evaluator.total_cost))

    model.to(device)
    model.eval()

    train_cm, train_costcm, train_costcm_norm, cost_per_step, logs, cost_per_signal_level, stats = evaluate_model(
        model, dl, batch_first, device, path_recorder, cost_evaluator, None, None, config['n_classes'])

    logs = dict((k, v.value()[0]) for k, v in logs.items() if isinstance(v, meter.AverageValueMeter))

    if use_acc_stats:
        predictions = torch.softmax(stats['preds'], dim=2)
        timed_predictions = [(predictions[i].squeeze().cpu().numpy(), i * config['frame_stride'] * 10) for i in
                             range(predictions.size(0))]

        accuracy_evaluator = StreamingAccuracy(label_names)
        acc_stats = accuracy_evaluator.compute_accuracy(timed_predictions, ds.timed_labels, up_to_time_ms=len_ms)
    else:
        acc_stats = None

    if return_pred_stats:
        stats['ys'] = ys
        return start_chkpt, logs, acc_stats, stats
    else:
        return start_chkpt, logs, acc_stats
