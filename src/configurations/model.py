import logging

import torch

from src.commons.pytorch.loading import get_state
from src.commons.sn.implementation.DenseResnet import DenseResnet
from src.commons.sn.implementation.ThreeDimNeuralFabric import ThreeDimNeuralFabric
from src.models.AdaptiveModel import AdaptiveModel
from src.models.KWS_model import kws_cnn_factory

logger = logging.getLogger(__name__)


def basic_config():
    static = False
    deter_eval = True
    arch_bias = 3

    start_chkpt = None


def rnn():
    recurrent_model = 'rnn'
    hidden_size = 100


def gru():
    recurrent_model = 'gru'
    hidden_size = 100


def kwscnn():
    model = 'kwscnn'
    kws_model = 'CNN_TRAD_FPOOL3'
    full = False

    sep_features = False


def cnf():
    model = 'cnf'
    n_layer = 2
    n_block = 1
    n_scale = 5
    n_chan = 32

    rounding_method = 'ceil'


def dresnet():
    model = 'dresnet'

    layers = 1
    blocks_per_layer = 3
    n_chan = 45
    shortcuts = False
    shortcuts_res = False
    shift = False
    bottlnecks = False
    bn_factor = 4
    bias = True
    dilatation = False
    pool_in = True


def get_rnn_model(input_size, recurrent_model, hidden_size):
    rnn_models = {
        'rnn': torch.nn.RNNCell,
        'gru': torch.nn.GRUCell
    }

    return rnn_models[recurrent_model](input_size=input_size, hidden_size=hidden_size)


def create_kwscnn(n_classes, static, deter_eval, input_dim, full, arch_bias, kws_model, recurrent_model, hidden_size,
                  sep_features, **kwargs):
    if static:
        static_proba = 1
    else:
        static_proba = -1

    stochastic_model = kws_cnn_factory(in_size=input_dim, n_classes=n_classes,
                                       static_node_proba=static_proba, full=full,
                                       model=kws_model, deter_eval=deter_eval, sep_features=sep_features)

    model_feats_dim = stochastic_model.features_dim

    rnn = get_rnn_model(input_size=model_feats_dim, recurrent_model=recurrent_model, hidden_size=hidden_size)

    model = AdaptiveModel(stochastic_model=stochastic_model, rnn=rnn, static=static, arch_bias=arch_bias)

    logger.info('# Global Model\n{}'.format(model))
    logger.info('# Stochastic Model\n{}'.format(model.stochastic_model))

    return model


def create_cnf(n_classes, static, deter_eval, arch_bias, n_layer, n_block, n_chan, n_scale, input_dim,
               rounding_method, recurrent_model, hidden_size, **kwargs):
    if static:
        static_proba = 1
    else:
        static_proba = -1

    stochastic_model = ThreeDimNeuralFabric(n_layer, n_block, n_chan, input_dim, n_classes, static_proba,
                                            n_scale=n_scale, rounding_method=rounding_method, deter_eval=deter_eval)

    # model_feats_dim = stochastic_model.features_dim
    model_feats_dim = stochastic_model.n_features

    rnn = get_rnn_model(input_size=model_feats_dim, recurrent_model=recurrent_model, hidden_size=hidden_size)

    model = AdaptiveModel(stochastic_model=stochastic_model, rnn=rnn, static=static, arch_bias=arch_bias)

    logger.info('# Global Model\n{}'.format(model))
    logger.info('# Stochastic Model\n{}'.format(model.stochastic_model))

    return model


def create_dresnet(layers, blocks_per_layer, n_chan, shortcuts, shortcuts_res, shift, input_dim, n_classes, static,
                   bottlnecks, bn_factor, bias, dilatation, pool_in, deter_eval, arch_bias, recurrent_model,
                   hidden_size, **kwargs):
    if static:
        static_proba = 1
    else:
        static_proba = -1

    stochastic_model = DenseResnet(layers=layers, blocks_per_layer=blocks_per_layer, n_channels=n_chan,
                                   shortcuts=shortcuts,
                                   shortcuts_res=shortcuts_res, shift=shift, input_dim=input_dim, n_classes=n_classes,
                                   static_node_proba=static_proba, bottlnecks=bottlnecks, bn_factor=bn_factor,
                                   bias=bias, dilatation=dilatation, pool_in=pool_in, deter_eval=deter_eval)

    # model_feats_dim = stochastic_model.features_dim
    model_feats_dim = stochastic_model.n_features

    rnn = get_rnn_model(input_size=model_feats_dim, recurrent_model=recurrent_model, hidden_size=hidden_size)

    model = AdaptiveModel(stochastic_model=stochastic_model, rnn=rnn, static=static, arch_bias=arch_bias)

    logger.info('# Global Model\n{}'.format(model))
    logger.info('# Stochastic Model\n{}'.format(model.stochastic_model))

    return model


models = {
    'kwscnn': create_kwscnn,
    'cnf': create_cnf,
    'dresnet': create_dresnet
}


def create_model(model, n_classes, start_chkpt, mongo_config_path, **kwargs):
    created_model = models[model](n_classes=n_classes, **kwargs)

    if start_chkpt is not None:
        logger.info("Loading state ...")
        state_dict = get_state(start_chkpt, mongo_config_path)
        created_model.loss = torch.nn.CrossEntropyLoss(weight=torch.zeros([n_classes]), reduction='none',
                                                       ignore_index=-7)
        created_model.load_state_dict(state_dict)
    return created_model


def model_config(ex):
    ex.config(basic_config)

    ex.named_config(rnn)
    ex.named_config(gru)

    ex.named_config(kwscnn)
    ex.named_config(cnf)
    ex.named_config(dresnet)

    models['kwscnn'] = ex.capture(create_kwscnn)
    models['cnf'] = ex.capture(create_cnf)
    models['dresnet'] = ex.capture(create_dresnet)

    create_model_capt = ex.capture(create_model)

    return create_model_capt
