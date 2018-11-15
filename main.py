import collections
import logging
import sys
import traceback
from collections import defaultdict, OrderedDict

import logger as data_logger
import numpy as np
from sacred.observers import FileStorageObserver

from src.commons.plotly.utils import get_traceopts_errors
from src.commons.pytorch.evaluation.evaluation import evaluate_model
from src.commons.sacred.oberservers.log import LogObserver
from src.commons.sn.implementation.ComputationCostEvaluator import ComputationCostEvaluator
from src.commons.sn.interface.PathRecorder import PathRecorder
from src.commons.utils import EMA, print_properties, save_checkpoint
from src.configurations.optim import optim_config
from src.results.utils import paretize_exp

logging.basicConfig(level=logging.INFO)

import torch
from sacred import Experiment
from sacred import SETTINGS as SACRED_SETTINGS
from visdom import Visdom

from src.commons import external_resources as external
from src.configurations.dataset import dataset_config
from src.configurations.model import model_config

logger = logging.getLogger(__name__)

ex = Experiment('ASC')


@ex.config
def config_exp():
    # seed = 1234
    device = 'cuda'
    nepochs = 200000
    lambda_reward = 0
    r_beta = .9
    r_gamma = 0
    debug = False

    use_visdom = True
    use_mongo = True

    if use_visdom:
        visdom_config_path = None
        visdom_conf = external.get_visdom_conf(visdom_config_path)
        ex.observers.append(LogObserver.create(visdom_conf))
    else:
        visdom_conf = None

    if use_mongo:
        mongo_config_path = './resources/mongo_credentials.json'
        ex.observers.append(external.get_mongo_obs(mongo_path=mongo_config_path))
    else:
        mongo_config_path = None
        ex_path = './runs'
        ex.observers.append(FileStorageObserver.create(ex_path))


create_dataset = dataset_config(ex)
create_model = model_config(ex)
create_optim = optim_config(ex)

torch.backends.cudnn.benchmark = True


def plot_(visdom, per_node_step_vals, node_names, title, win, log_func=None):
    for node_idx, node_name in enumerate(node_names):
        per_step_vals = per_node_step_vals[node_idx]
        plot_meters(visdom, per_step_vals, node_name, title, win, log_func=log_func)


def plot_meters(visdom, values, name, title=None, win=None, error_bars=True, log_func=None):
    x, means, stds = zip(*([(k, *v.value()) for k, v in sorted(values.items())]))
    opts = dict(showlegend=True)
    if error_bars:
        errors = np.array(stds)
        errors[errors == np.inf] = 0
        opts['traceopts'] = get_traceopts_errors(name, errors.tolist())
    if title is not None:
        opts['title'] = title
    if log_func is not None:
        log_func(name + '.x', list(x))
        log_func(name + '.means', list(means))
        log_func(name + '.stds', list(stds))

    win = win or name
    update = 'replace' if visdom.win_exists(win) else None

    try:
        visdom.line(Y=np.array(means), X=np.array(x), name=name, win=win, update=update, opts=opts)
    except ConnectionError as err:
        logger.warning('#####\n#####\n#####')
        logger.warning('Problem when plotting win:{}, title:{} for node {}'.format(win, title, name))
        logger.warning(err)
        traceback.print_exc()
        logger.warning('#####\n#####\n#####')
        return False
    return True


def model_name(config):
    model = config['model']
    if model == 'cnf':
        model += '_l{}s{}c{}'.format(config['n_layer'], config['n_scale'], config['n_chan'])

    return model


def format_exp_name(_id, config):
    name = model_name(config)
    if config['static']:
        name += '_static'
    name += '_{}'.format(_id)

    if config['debug']:
        name = 'Debug_' + name

    return name


@ex.main
def main(_run, nepochs, device, use_visdom, visdom_conf, n_classes, lambda_reward, r_beta, r_gamma, _config):
    exp_name = format_exp_name(_run._id, _config)
    if use_visdom:
        visdom_conf.update(env=exp_name)
        _run.info['visdom_server'] = "{server}:{port}/env/{env}".format(**visdom_conf)
    else:
        _run.info['visdom_server'] = "No visdom"

    _run.info['exp_name'] = exp_name
    front = _run.info['front'] = {}

    xp_logger = data_logger.Experiment(exp_name, use_visdom=use_visdom, visdom_opts=visdom_conf, time_indexing=False,
                                       xlabel='Epoch', log_git_hash=False)
    xp_logger.add_log_hook(_run.log_scalar)
    if use_visdom:
        xp_logger.plotter.windows_opts = defaultdict(lambda: dict(showlegend=True))

    viz = Visdom(**visdom_conf) if use_visdom else None

    # Dataset creation
    logger.info('### Dataset ###')

    ds, batch_first, class_w = create_dataset()
    _run.info['class_weights'] = class_w.tolist()

    confusion_matrix_opts = {
        'columnnames': ds['train'].dataset.ordered_class_names,
        'rownames': ds['train'].dataset.ordered_class_names}

    # Model Creation
    logger.info('### Model ###')

    adaptive_model = create_model()
    adaptive_model.loss = torch.nn.CrossEntropyLoss(weight=class_w, reduction='none', ignore_index=-7)

    path_recorder = PathRecorder(adaptive_model.stochastic_model)
    cost_evaluator = ComputationCostEvaluator(node_index=path_recorder.node_index, bw=False)
    # cost_evaluator = SimpleEdgeCostEvaluator(node_index=path_recorder.node_index, bw=False)

    cost_evaluator.init_costs(adaptive_model.stochastic_model)
    logger.info('Cost: {:.5E}'.format(cost_evaluator.total_cost))

    adaptive_model.to(device)

    # Optim Creation
    logger.info('### Optim ###')
    optimizer, schedulder = create_optim(params=adaptive_model.get_param_groups())

    # Check the param_groups order, to be sure to get the learning rates in the right order for logging
    assert [pg['name'] for pg in optimizer.param_groups] == ['arch_params', 'pred_params']

    def optim_closure(loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Logger creation
    splits = ['train', 'validation', 'test']
    metrics = ['classif_loss', 'arch_loss', 'reward', 'lambda_reward', 'silence_ratio', 'accuracy', 'average_cost',
               'learning_rate_pred', 'learning_rate_arch']

    for split in splits:
        xp_logger.ParentWrapper(tag=split, name='parent'.format(split),
                                children=[xp_logger.SimpleMetric(name=metric) for metric in metrics])

    train_cost_loggers = dict((i, xp_logger.AvgMetric(name='train_cost', tag=name))
                              for i, name in enumerate(ds['train'].dataset.ordered_class_names))
    train_cost_loggers_perc = dict((i, xp_logger.AvgMetric(name='train_cost_perceived', tag=name))
                                   for i, name in enumerate(ds['train'].dataset.ordered_class_names))

    node_names = adaptive_model.stochastic_model.ordered_node_names
    # entropy_loggers = [xp_logger.SimpleMetric(name='entropy', tag=name) for name in node_names]
    entropy_loggers = OrderedDict(
        (i, xp_logger.SimpleMetric(name='entropy_per_node', tag=name)) for i, name in enumerate(node_names))
    # proba_loggers = [xp_logger.SimpleMetric(name='proba', tag=name) for name in node_names]
    proba_loggers = OrderedDict(
        (i, xp_logger.SimpleMetric(name='proba_per_node', tag=name)) for i, name in enumerate(node_names))

    val_cost_loggers = dict((i, xp_logger.AvgMetric(name='val_cost', tag=name))
                            for i, name in enumerate(ds['validation'].dataset.ordered_class_names))
    val_cost_loggers_perc = dict((i, xp_logger.AvgMetric(name='val_cost_perceived', tag=name))
                                 for i, name in enumerate(ds['validation'].dataset.ordered_class_names))

    test_cost_loggers = dict((i, xp_logger.AvgMetric(name='test_cost', tag=name))
                             for i, name in enumerate(ds['test'].dataset.ordered_class_names))
    test_cost_loggers_perc = dict((i, xp_logger.AvgMetric(name='test_cost_perceived', tag=name))
                                  for i, name in enumerate(ds['test'].dataset.ordered_class_names))

    if use_visdom:
        print_properties(viz, _config)
        print_properties(viz, _run.info)

    ema_reward = EMA(r_beta)  # Init the exponential moving average
    for n in range(1, nepochs + 1):
        logger.info('### Sarting epoch n°{} ### {}'.format(n, _run.info['visdom_server']))
        logger.info(' '.join(sys.argv))

        if schedulder:
            schedulder.step(n)
            arch_lr, pred_lr = schedulder.get_lr()
            xp_logger.Parent_Train.update(learning_rate_pred=pred_lr, learning_rate_arch=arch_lr)

        # Training
        adaptive_model.train()
        train_cm, train_costcm, train_costcm_norm, train_cost_per_step, logs, train_cost_per_signal_level, train_stats = evaluate_model(
            adaptive_model,
            ds['train'],
            batch_first, device,
            path_recorder,
            cost_evaluator,
            train_cost_loggers,
            train_cost_loggers_perc,
            n_classes,
            lambda_reward,
            ema_reward, r_gamma,
            optim_closure,
            name='Train')

        xp_logger.Parent_Train.update(**dict((k, v.value()[0]) for k, v in logs.items()))

        for node_idx, ent in train_stats['en'].items():
            entropy_loggers[node_idx].update(ent.value()[0])

        for node_idx, prob in train_stats['pn'].items():
            proba_loggers[node_idx].update(prob.value()[0])

        # Evaluation
        adaptive_model.eval()
        val_cm, val_costcm, val_costcm_norm, val_cost_per_step, logs, val_cost_per_signal_level, val_stats = evaluate_model(
            adaptive_model,
            ds['validation'],
            batch_first, device,
            path_recorder,
            cost_evaluator,
            val_cost_loggers,
            val_cost_loggers_perc,
            n_classes, lambda_reward,
            ema_reward, r_gamma,
            name='Validation')

        xp_logger.Parent_Validation.update(**dict((k, v.value()[0]) for k, v in logs.items()))

        test_cm, test_costcm, test_costcm_norm, test_cost_per_step, logs, test_cost_per_signal_level, test_stats = evaluate_model(
            adaptive_model,
            ds['test'], batch_first,
            device,
            path_recorder,
            cost_evaluator,
            test_cost_loggers,
            test_cost_loggers_perc,
            n_classes, lambda_reward,
            ema_reward, r_gamma,
            name='Test')
        xp_logger.Parent_Test.update(**dict((k, v.value()[0]) for k, v in logs.items()))

        if use_visdom:
            # Log
            plot_(viz, train_stats['es'], node_names, f'Entropy per step {n} - Train', 'train_eps',
                  log_func=_run.log_scalar)
            plot_(viz, train_stats['ps'], node_names, f'Probability per step {n} - Train', 'train_pps',
                  log_func=_run.log_scalar)
            try:
                viz.heatmap(train_cm, win='train_cm',
                            opts={**confusion_matrix_opts, 'title': 'Train Confusion matrix'})
                viz.heatmap(val_cm, win='val_cm', opts={**confusion_matrix_opts, 'title': 'Val Confusion matrix'})
                viz.heatmap(test_cm, win='test_cm',
                            opts={**confusion_matrix_opts, 'title': 'Test Confusion matrix'})

                # viz.heatmap(train_costcm, win='train_cost_matrix',
                #             opts={**confusion_matrix_opts, 'title': 'Train cost matrix'})
                # viz.heatmap(val_costcm, win='val_cost_matrix', opts={**confusion_matrix_opts, 'title': 'Val cost matrix'})
                # viz.heatmap(test_costcm, win='test_cost_matrix',
                #             opts={**confusion_matrix_opts, 'title': 'Test cost matrix'})

                viz.heatmap(train_costcm_norm, win='train_cost_matrix_norm',
                            opts={**confusion_matrix_opts, 'title': 'Train cost matrix Normalized'})
                viz.heatmap(val_costcm_norm, win='val_cost_matrix_norm',
                            opts={**confusion_matrix_opts, 'title': 'Val cost matrix Normalized'})
                viz.heatmap(test_costcm_norm, win='test_cost_matrix_norm',
                            opts={**confusion_matrix_opts, 'title': 'Test cost matrix Normalized'})

            except ConnectionError as err:
                logger.warning('Error in heatmaps:')
                logger.warning(err)
                traceback.print_exc()

            plot_meters(viz, train_cost_per_step, 'train_cps', 'Cost per step {}'.format(n), win='cps',
                        log_func=_run.log_scalar)
            plot_meters(viz, val_cost_per_step, 'val_cps', win='cps', log_func=_run.log_scalar)
            plot_meters(viz, test_cost_per_step, 'test_cps', win='cps', log_func=_run.log_scalar)

            plot_meters(viz, train_cost_per_signal_level, 'cost/sig_train', 'Cost per signal {}'.format(n), win='cpsig',
                        error_bars=False, log_func=_run.log_scalar)
            plot_meters(viz, val_cost_per_signal_level, 'cost/sig_val', win='cpsig', error_bars=False,
                        log_func=_run.log_scalar)
            plot_meters(viz, test_cost_per_signal_level, 'cost/sig_test', win='cpsig', error_bars=False,
                        log_func=_run.log_scalar)

        xp_logger.log_with_tag(tag='*', reset=True)

        msg = 'Losses: {:.3f}({:.3E})-{:.3f}-{:.3f}, Accuracies: {:.3f}-{:.3f}-{:.3f}, Avg cost: {:.3E}-{:.3E}-{:.3E}'
        msg = msg.format(xp_logger.classif_loss_train, xp_logger.reward_train, xp_logger.classif_loss_validation,
                         xp_logger.classif_loss_test, xp_logger.accuracy_train, xp_logger.accuracy_validation,
                         xp_logger.accuracy_test, xp_logger.average_cost_train, xp_logger.average_cost_validation,
                         xp_logger.average_cost_test)
        logger.info(msg)

        pareto_data = {'cost': xp_logger.logged['average_cost_validation'].values(),
                       'acc': xp_logger.logged['accuracy_validation'].values(),
                       '_orig_': xp_logger.logged['average_cost_validation'].keys()}

        pareto = paretize_exp(pareto_data, x_name='cost', crit_name='acc')

        if n in pareto['_orig_']:
            logger.info('New on front !')
            front.update(**pareto)
            save_checkpoint(adaptive_model, ex, n)
        elif n > 0 and n % 50 == 0:
            logger.info('Checkpointing')
            save_checkpoint(adaptive_model, ex, n)

        logger.info(pareto['_orig_'])
        best_epoch = pareto['_orig_'][-1]
        logger.info('Best \tVal: {:.3f} - Test: {:.3f} (Epoch {})\n'
                    .format(xp_logger.logged['accuracy_validation'][best_epoch],
                            xp_logger.logged['accuracy_test'][best_epoch], best_epoch))


if __name__ == '__main__':
    ex.run_commandline()
