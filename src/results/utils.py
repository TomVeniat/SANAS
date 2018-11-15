import csv
import logging
import os
import re
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def paretize_exp(data, x_name, crit_name, value_name=None):
    if value_name is None:
        value_name = crit_name

    final_x = []
    final_crit = []
    final_vals = []
    final_origins = []
    cur_best_crit = None
    cur_best_x = None
    for x, crit, val, orig in sorted(zip(data[x_name], data[crit_name], data[value_name], data['_orig_'])):
        if len(final_x) == 0 or crit > cur_best_crit:
            cur_best_crit = crit
            if x == cur_best_x:
                final_crit[-1] = crit
                final_vals[-1] = val
                final_origins[-1] = orig
            else:
                final_x.append(x)
                final_crit.append(crit)
                final_vals.append(val)
                final_origins.append(orig)
            cur_best_x = x
    return {x_name: final_x,
            crit_name: final_crit,
            value_name: final_vals,
            '_orig_': final_origins}
