# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import numpy as np
from ipdb import set_trace
from sklearn.metrics import recall_score, precision_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not dist_utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def egomcq_accuracy_metrics(preds, labels, types, group_list = ["Inter-video", "Intra-video"]):
    metrics = {}
    type_list = torch.unique(types)
    # group_list = ["Intra-video", "Inter-video"]
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, typer in zip(preds, labels, types):
            if typer == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct/total
        metrics[group_i] = accuracy * 100
    return metrics



def get_marginal_indexes(actions, mode):
    """For each verb/noun retrieve the list of actions containing that verb/name
        Input:
            mode: "verb" or "noun"
        Output:
            a list of numpy array of indexes. If verb/noun 3 is contained in actions 2,8,19,
            then output[3] will be np.array([2,8,19])
    """
    vi = []
    for v in range(actions[mode].max()+1):
        vals = actions[actions[mode] == v].index.values
        if len(vals) > 0:
            vi.append(vals)
        else:
            vi.append(np.array([0]))
    return vi


def marginalize(probs, indexes):
    mprobs = []
    for ilist in indexes:
        mprobs.append(probs[:, ilist].sum(1))
    return np.array(mprobs).T

def calculate_DCG(similarity_matrix, relevancy_matrix, k_counts):
    """
    Calculates the Discounted Cumulative Gain (DCG) between two modalities for
    the first modality.
    DCG = \sum_{i=1}^k \frac{rel_i}{log_2(i + 1)}
    i.e. the sum of the k relevant retrievals which is calculated as the scaled
    relevancy for the ith item. The scale is designed such that early
    retrievals are more important than later retrievals.
    Params:
        - similarity_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality. The [ith,jth] element is the predicted similarity
          between the ith item from the first modality and the jth item from
          the second modality.
        - relevancy_matrix: matrix of size n1 x n2 (see similarity_matrix
          above). The [ith, jth] element is the semantic relevancy between the
          ith item from the first modality and the jth item from the second
          modality.
        - k_counts: matrix of size n1 x n2 (see similarity_matrix above) which
          includes information on which items to use to calculate the DCG for
          (see calculate_k_counts for more info on this matrix).
    Returns:
        - The DCG for each item in the first modality, a n1 length vector.
    """
    x_sz, y_sz = similarity_matrix.shape
    ranks = np.argsort(similarity_matrix)[:, ::-1]
    # Create vector of size (n,) where n is the length of the last dimension in
    # similarity matrix
    # This vector is of the form log(i+1)
    logs = np.log2(np.arange(y_sz) + 2)
    # Convert logs into the divisor for the DCG calculation, of size similarity
    # matrix
    divisors = np.repeat(np.expand_dims(logs, axis=0), x_sz, axis=0)

    # mask out the sorted relevancy matrix to only use the first k relevant
    # retrievals for each item.
    columns = np.repeat(np.expand_dims(np.arange(x_sz), axis=1), y_sz, axis=1)
    numerators = relevancy_matrix[columns, ranks] * k_counts
    # Calculate the final DCG score (note that this isn't expected to sum to 1)
    return np.sum(numerators / divisors, axis=1)


def calculate_k_counts(relevancy_matrix):
    """
    Works out the maximum number of allowed retrievals when working out the
    Discounted Cumulative Gain. For each query the DCG only uses the first k
    items retrieved which constitute the k relevant items for that query
    (otherwise the nDCG scores can be deceptively high for bad rankings).
    Params:
        - relevancy_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality.  The [ith, jth] element is the semantic relevancy
          between the ith item from the first modality and the jth item from
          the second modality.
    Returns:
        - Matrix of size n1 x n2 (see relevancy matrix for more info). This is
          created as a mask such that if the [ith, jth] element is 1 it
          represents a valid item to use for the calculation of DCG for the
          ith item after sorting. For example, if relevancy matrix of:
        [[1, 0.5, 0],
          [0, 0  , 1]]
          is given, then the k_counts matrix will be:
        [[1, 1, 0],
         [1, 0, 0]]
         i.e. the first row has 2 non-zero items, so the first two retrieved
         items should be used in the calculation. In the second row there is
         only 1 relevant item, therefore only the first retrieved item should
         be used for the DCG calculation.
    """
    return (np.sort(relevancy_matrix)[:, ::-1] > 0).astype(int)


def calculate_IDCG(relevancy_matrix, k_counts):
    """
    Calculates the Ideal Discounted Cumulative Gain (IDCG) which is the value
    of the Discounted Cumulative Gain (DCG) for a perfect retrieval, i.e. the
    items in the second modality were retrieved in order of their descending
    relevancy.
    Params:
        - relevancy_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality. The [ith, jth] element is the semantic relevancy
          between the ith item from the first modality and the jth item from
          the second modality.
        - k_counts: matrix of size n1 x n2 (see similarity_matrix above) which
          includes information on which items to use to calculate the DCG for
          (see calculate_k_counts for more info on this matrix).
    """
    return calculate_DCG(relevancy_matrix, relevancy_matrix, k_counts)


def calculate_nDCG(similarity_matrix, relevancy_matrix, k_counts=None, IDCG=None, reduction='mean'):
    """
    Calculates the normalised Discounted Cumulative Gain (nDCG) between two
    modalities for the first modality using the Discounted Cumulative Gain
    (DCG) and the Ideal Discounted Cumulative Gain (IDCG).
    nDCG = \frac{DCG}{IDCG}
    Params:
        - similarity_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the second
          modality. The [ith,jth] element is the predicted similarity between
          the ith item from the first modality and the jth item from the second
          modality.
        - relevancy_matrix: matrix of size n1 x n2 (see similarity_matrix
          above). The [ith, jth] element is the semantic relevancy between the
          ith item from the first modality and the jth item from the second
          modality.
        - k_counts: optional parameter: matrix of size n1 x n2 (see
          similarity_matrix above) which includes information on which items to
          use to calculate the DCG for (see calculate_k_counts for more info on
          this matrix). This will be calculated using calculate_IDCG if not
          present, but should be pre-processed for efficiency.
        - IDCG: Optional parameter which includes the pre-processed Ideal
          Discounted Cumulative Gain (IDCG). This is a vector of size n1 (see
          similarity_matrix above) which contains the IDCG value for each item
          from the first modality. This will be calculated using calculate_IDCG
          if not present, but should be pre-processed for efficiency.
        - reduction: what to use to reduce the different nDCG scores. By
          default this applies np.mean across all different queries.
    Returns:
        - The nDCG values for the first modality.
    """
    if k_counts is None:
        k_counts = calculate_k_counts(relevancy_matrix)
    DCG = calculate_DCG(similarity_matrix, relevancy_matrix, k_counts)
    if IDCG is None:
        IDCG = calculate_IDCG(relevancy_matrix, k_counts)
    if reduction == 'mean':
        return np.mean(DCG / IDCG)
    elif reduction is None:
        return DCG / IDCG


def calculate_mAP(sim_mat, relevancy_matrix):
    """
    Computes the mean average precision according to the following formula of
    average precision:
    \frac{\sum_{k=1}^n p(k) x rel(k)}{num_rel_docs}
    where p(k) is the precision at k, rel(k) is an indicator function
    determining whether the kth returned item is relevant or not and
    num_rel_docs is the number of relevant items to find within the search.
    The mean average precision is the mean of the average precision for each
    query item (i.e row in the matrix)
    This function takes in two parameters:
        - sim_mat: a NxM matrix which represents the similarity between two
        modalities (with modality 1 being of size N and modality 2 of size M).
        - relevancy_matrix: an NxM matrix which represents the relevancy between two
        modalities of items (with modality 1 being of size N and modality 2 of
        size M).
    """
    # Find the order of the items in modality 2 according to modality 1
    ranked_order = (-sim_mat).argsort()
    ranked_sim_mat = sim_mat[np.arange(sim_mat.shape[0])[:, None], ranked_order]
    # re-order the relevancy matrix to accommodate the proposals
    ranked_rel_mat = relevancy_matrix[np.arange(relevancy_matrix.shape[0])[:, None], ranked_order]

    # find the number of relevant items found at each k
    cumulative_rel_mat = np.cumsum(ranked_rel_mat, axis=1)
    # Mask this ensuring that it is non zero if the kth term is 1 (rel(k) above)
    cumulative_rel_mat[ranked_rel_mat != 1] = 0
    # find the divisor for p(k)
    divisor = np.arange(ranked_rel_mat.shape[1]) + 1

    # find the number of relevant docs per query item
    number_rel_docs = np.sum(ranked_rel_mat == 1, axis=1)

    # find the average precision per query, within np.sum finds p(k) * rel(k)
    avg_precision = np.sum(cumulative_rel_mat / divisor, axis=1) / number_rel_docs
    mAP = np.mean(avg_precision)
    return mAP


def get_mAP(similarity_matrix, rel_matrix):
    vis_map = calculate_mAP(similarity_matrix, rel_matrix)
    txt_map = calculate_mAP(similarity_matrix.T, rel_matrix.T)
    return vis_map, txt_map, (vis_map + txt_map) / 2


def get_nDCG(similarity_matrix, rel_matrix):
    vis_k_counts = calculate_k_counts(rel_matrix)
    txt_k_counts = calculate_k_counts(rel_matrix.T)
    vis_IDCG = calculate_IDCG(rel_matrix, vis_k_counts)
    txt_IDCG = calculate_IDCG(rel_matrix.T, txt_k_counts)
    vis_nDCG = calculate_nDCG(similarity_matrix, rel_matrix, k_counts=vis_k_counts, IDCG=vis_IDCG)
    txt_nDCG = calculate_nDCG(similarity_matrix.T, rel_matrix.T, k_counts=txt_k_counts, IDCG=txt_IDCG)
    return vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_recall(output, target):
    pred = output.max(1)[1].cpu().numpy()
    target = target.cpu().numpy()
    return recall_score(target, pred)

def get_mean_accuracy(cm):
    list_acc = []
    for i in range(len(cm)):
        acc = 0
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
        list_acc.append(acc)

    return 100 * np.mean(list_acc), 100 * np.trace(cm) / np.sum(cm)


def compute_map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return compute_map(fix, gt_array)


def create_submission(video_list, predictions, out_file):
    assert len(video_list) == predictions.shape[0]
    with open(out_file, 'w') as f:
        for i, video_id in enumerate(video_list):
            pred_str = ' '.join(map(lambda x: str(x), predictions[i].tolist()))
            f.write('{} {}\n\n'.format(video_id, pred_str))


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics
