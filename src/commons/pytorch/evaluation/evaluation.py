from collections import defaultdict

import torch
from torchnet import meter
from tqdm import tqdm

from src.commons.utils import compute_discounted_rewards, compute_entropy, print_batch


def update_matrices(cost_mat, conf_mat, preds, truth, costs):
    # print(f'cost_mat: {cost_mat.shape}')
    # print(f'conf_mat: {conf_mat.shape}')
    # print(f'preds: {preds.shape}')
    # print(f'truth: {truth.shape}')
    # print(f'costs: {costs.shape}')
    for t, p, c in zip(truth, preds, costs):
        cost_mat[t.item(), p.item()] += c.item()
        conf_mat[t, p] += 1


def log_costs(costs, valid_mask, cost_per_step, cost_per_label, cost_per_signal_level, signal_levels, y):
    """

    :param costs: seq_len * batch_size
    :param valid_mask: seq_len * batch_size
    :param cost_per_step: dict of loggers, where the key is the timestep index and the value the associated logger
    :param cost_per_label: dict of loggers. Keys are the label idx (int), and valeus are corresponding loggers
    :param y: seq_len * batch_size, corresponds to the label (int) of each sample
    :return:
    """
    # assert valid_mask.sum().item() == valid_mask.numel()
    # todo remove the usage of valid_mask

    #  Update per-timestep statistics
    for j, (t, valid) in enumerate(zip(costs.split(1, 0), valid_mask.split(1, 0))):
        step_costs = t[valid]
        for elt in step_costs:
            cost_per_step[j].add(elt.item())

    #  Update per-class statistics
    for label, logger in cost_per_label.items():
        label_mask = (y == label).cpu()
        n_samples = label_mask.sum().item()
        if n_samples > 0:
            logger.update(costs[label_mask].mean(), n=n_samples)

    # Compute cost per signal
    signal_levels = list(signal_levels)
    for i, seq in enumerate(signal_levels):
        for j, step_signal in enumerate(seq):
            cost_per_signal_level[step_signal.item()].add(int(costs[j, i].item()))


def log_model_stats(probabilities, valid_mask, proba_per_node, entropy_per_node, proba_per_step, entropy_per_step):
    """

    :param probabilities: seq_len*b_size*n_node
    :param valid_mask: seq_len*b_size
    :param proba_per_node:
    :param entropy_per_node:
    :return:
    """
    n_node = probabilities.size(-1)
    entropies = compute_entropy(probabilities)
    for step in range(probabilities.size(0)):
        n_valid = valid_mask[step].sum().item()
        # Probabilities of the nodes for the valid samples of this timestep, size=n_valid*n_node
        step_valid_probas = probabilities[step][valid_mask[step]]
        step_valid_entropies = entropies[step][valid_mask[step]]
        for valid_idx in range(n_valid):
            # The proba vector for one valid timestep, size=n_node
            valid_probas = step_valid_probas[valid_idx]
            valid_entropies = step_valid_entropies[valid_idx]
            for node_idx in range(n_node):
                proba_per_step[node_idx][step].add(valid_probas[node_idx].item())
                entropy_per_step[node_idx][step].add(valid_entropies[node_idx].item())

                proba_per_node[node_idx].add(valid_probas[node_idx].item())
                entropy_per_node[node_idx].add(valid_entropies[node_idx].item())


def evaluate_model(adaptive_model, dataloader, batch_first, device, path_recorder, cost_evaluator, cost_per_label,
                   cost_per_perceived_label, n_classes, lambda_reward=None, reward_bl=None, reward_gamma=None,
                   optim_closure=None, name=None, return_preds=False):
    confusion_matrix = torch.zeros(n_classes, n_classes)
    cost_matrix = torch.zeros(n_classes, n_classes)

    cost_per_step = defaultdict(meter.AverageValueMeter)
    cost_per_signal_level = defaultdict(meter.AverageValueMeter)
    proba_per_node = defaultdict(meter.AverageValueMeter)
    entropy_per_node = defaultdict(meter.AverageValueMeter)
    proba_per_step = defaultdict(lambda: defaultdict(meter.AverageValueMeter))
    entropy_per_step = defaultdict(lambda: defaultdict(meter.AverageValueMeter))

    logs = defaultdict(meter.AverageValueMeter)

    for i, (images, labels, infos) in enumerate(tqdm(dataloader, desc=name)):
        x = images.to(device)
        y = labels.to(device)

        with torch.set_grad_enabled(adaptive_model.training):
            predictions = adaptive_model(x, batch_first)

        if batch_first:
            y = y.transpose(0, 1).contiguous()  # Get a seq-first y
            predictions = predictions.transpose(0, 1).contiguous()  # Get a seq-first predictions

        per_sample_loss, n_correct, y_hat, valid_mask = get_loss_acc(predictions, y, adaptive_model.loss)

        n_valid = valid_mask.sum().item()
        classif_loss = per_sample_loss.sum() / n_valid

        sampled, pruned = path_recorder.get_architectures(adaptive_model.out_nodes)
        costs_s = cost_evaluator.get_costs(sampled)  # Sampled cost
        costs_p = cost_evaluator.get_costs(pruned)  # Pruned cost

        # print(set([e.item() for e in costs_p.squeeze()]))

        signal_per_frame = map(lambda elt: (elt[0]['signal_per_frame'] * 100).round(), infos)

        if cost_per_label is not None:
            log_costs(costs_p, valid_mask, cost_per_step, cost_per_label, cost_per_signal_level, signal_per_frame, y)

        if cost_per_perceived_label is not None:
            log_costs(costs_p, valid_mask, cost_per_step, cost_per_perceived_label, cost_per_signal_level,
                      signal_per_frame, y_hat)

        log_model_stats(adaptive_model.probabilities, valid_mask, proba_per_node, entropy_per_node, proba_per_step,
                        entropy_per_step)

        update_matrices(cost_matrix, confusion_matrix, y_hat[valid_mask], y[valid_mask], costs_p[valid_mask])

        if (lambda_reward is not None) and (reward_bl is not None) and (reward_gamma is not None):
            reward = per_sample_loss + lambda_reward * costs_p.to(device)
            reward = -reward  # we want to maximize the reward
            mean_reward = reward.mean().item()

            if adaptive_model.training:
                reward_bl.update(mean_reward)

            reward -= reward_bl.val
            discounted_reward = compute_discounted_rewards(reward, reward_gamma)

            stacked_log_probas = torch.stack(adaptive_model.stochastic_model.batched_log_probas)
            architecture_loss = -stacked_log_probas[valid_mask] * discounted_reward[valid_mask].unsqueeze(-1)
            architecture_loss = architecture_loss.mean()

            if adaptive_model.training:
                optim_closure(loss=classif_loss + architecture_loss)

            logs['arch_loss'].add(architecture_loss.item())
            logs['reward'].add(mean_reward)
            logs['lambda_reward'].add(lambda_reward)

        logs['classif_loss'].add(classif_loss.item())
        logs['accuracy'].add(n_correct / n_valid)
        logs['average_cost'].add(costs_p[valid_mask.cpu()].sum().item(), n=n_valid)
        logs['silence_ratio'].add((y == 0).sum().item() / n_valid)

    m = confusion_matrix != 0
    cost_matrix_norm = cost_matrix.clone()
    cost_matrix_norm[m] /= confusion_matrix[m]

    return confusion_matrix, cost_matrix, cost_matrix_norm, cost_per_step, logs, cost_per_signal_level, \
           dict(pn=proba_per_node, en=entropy_per_node, ps=proba_per_step, es=entropy_per_step, preds=predictions, costs_p=costs_p)


def get_loss_acc(preds, y, loss_function):
    """
    :param preds: Seq_len x B x N_classes
    :param y: Target Tensor, size Seq_len x B
    :param loss_function: The loss function to apply
    :param cm: The confusion matrix to update
    :return: the mean loss per sample, the number of correct predictions and the number of valid samples
    """
    n_classes = preds.size(-1)
    valid_mask = (y != loss_function.ignore_index).cpu()

    flatten_pred = preds.view(-1, n_classes)
    flatten_y = y.view(-1)

    loss = loss_function(flatten_pred, flatten_y).view(y.size())
    loss[1 - valid_mask] = 0

    _, y_hat = torch.max(preds, 2)

    n_correct = (y_hat[valid_mask] == y[valid_mask]).sum().item()

    return loss, n_correct, y_hat, valid_mask

