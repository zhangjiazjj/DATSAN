import torch.nn as nn
import numpy as np
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve, auc

def calculate_measure(tp, fn, fp):
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return p, r, f1

class Measure(object):
    def __init__(self, num_classes, target_class):
        self.num_classes = num_classes
        self.target_class = target_class
        self.true_positives = {}
        self.false_positives = {}
        self.false_negatives = {}
        self.probabilities = {}  # 存储概率以计算AUC
        self.labels = {}  # 存储标签以计算AUC
        self.target_best_f1 = 0.0
        self.target_best_f1_epoch = 0
        self.reset_info()

    def reset_info(self):
        for cur_class in range(self.num_classes):
            self.true_positives[cur_class] = []
            self.false_positives[cur_class] = []
            self.false_negatives[cur_class] = []
            self.probabilities[cur_class] = []
            self.labels[cur_class] = []

    def append_measures(self, log_probabilities, labels):
        probabilities = torch.exp(log_probabilities)  # 将对数概率转换为概率
        predicted_classes = log_probabilities.argmax(dim=1)
        for cl in range(self.num_classes):
            cl_indices = labels == cl
            pos = predicted_classes == cl
            hits = predicted_classes[cl_indices] == labels[cl_indices]

            tp = hits.sum().item()
            fn = cl_indices.sum().item() - tp
            fp = pos.sum().item() - tp

            self.true_positives[cl].append(tp)
            self.false_negatives[cl].append(fn)
            self.false_positives[cl].append(fp)
            self.probabilities[cl].extend(probabilities[:, cl].tolist())
            self.labels[cl].extend((labels == cl).int().tolist())


    def get_each_timestamp_measure(self):
        macro_precisions, macro_recalls, macro_f1s, auc_scores = [], [], [], []
        num_timestamps = len(next(iter(self.true_positives.values())))
        for i in range(num_timestamps):
            sum_p = sum_r = sum_f1 = 0
            valid_classes = 0
            for cl in range(self.num_classes):
                tp = self.true_positives[cl][i]
                fn = self.false_negatives[cl][i]
                fp = self.false_positives[cl][i]
                p, r, f1 = calculate_measure(tp, fn, fp)
                if tp or fn or fp:
                    sum_p += p
                    sum_r += r
                    sum_f1 += f1
                    valid_classes += 1
            if valid_classes:
                sum_p /= valid_classes
                sum_r /= valid_classes
                sum_f1 /= valid_classes
            macro_precisions.append(sum_p)
            macro_recalls.append(sum_r)
            macro_f1s.append(sum_f1)

        return macro_precisions, macro_recalls, macro_f1s
    def get_total_measure(self):
        sum_p = sum_r = sum_f1 = 0
        valid_classes = 0
        tprs = {}
        fprs = {}
        roc_data = {}
        for cl in range(self.num_classes):
            tp = sum(self.true_positives[cl])
            fn = sum(self.false_negatives[cl])
            fp = sum(self.false_positives[cl])
            p, r, f1 = calculate_measure(tp, fn, fp)
            if tp or fn or fp:
                sum_p += p
                sum_r += r
                sum_f1 += f1
                valid_classes += 1

                if len(self.labels[cl]) > 1 and len(set(self.labels[cl])) > 1:
                    fpr, tpr, _ = roc_curve(self.labels[cl], self.probabilities[cl])
                    roc_auc = auc(fpr, tpr)
                    roc_data[cl] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        if valid_classes:
            sum_p /= valid_classes
            sum_r /= valid_classes
            sum_f1 /= valid_classes
        return sum_p, sum_r, sum_f1, roc_data[1]

    def update_best_f1(self, cur_f1, cur_epoch):
        if cur_f1 > self.target_best_f1:
            self.target_best_f1 = cur_f1
            self.target_best_f1_epoch = cur_epoch


class GeneralizedCELoss1(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss1, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = torch.mean(F.cross_entropy(logits, targets, reduction='none') * loss_weight)
        return loss

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    # Move the tensor to the CPU if necessary
    mx = mx.to('cpu')

    # Get the indices and values from the PyTorch sparse tensor
    indices = mx.indices().numpy()
    values = mx.val.numpy()

    # Convert to COO format
    mx = sp.coo_matrix((values, indices), shape=mx.shape)
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    adj = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj
