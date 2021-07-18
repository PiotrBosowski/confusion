import copy
import json

import numpy as np


class Confusion:
    """
    Class for storing confusion matrix and calculating its statistics.
    """

    # size: int
    # matrix: 2-D np.array
    # labels: 1-D np.array
    # groups: dict of groups for stratifying the data

    def __init__(self, labels, matrix=None, groups=None):
        # group is a list of labels you want to evaluate closer
        # (separately from the rest) or to compare with different group
        try:
            # in case labels is a list of labels
            # check for no duplicates
            assert len(labels) == len(set(labels))
            self.labels = [label for label in labels]
            self.size = len(self.labels)
        except Exception:
            # assuming labels is a number of labels
            assert type(labels) is int
            self.size = labels
            # assigning generic names:
            self.labels = map(str, list(range(self.size)))
        self.matrix = np.zeros(shape=(self.size, self.size), dtype=np.int)
        if matrix is not None:
            matrix = np.array(matrix)
            # error will be raised if dims dont match
            self.matrix += matrix
        self.groups = {}
        if groups is not None:
            # check if grouped labels exist
            for group_name, grouped_labels in groups.items():
                self.set_group(group_name, grouped_labels)

    def set_group(self, name, labels):
        # check if labels are contained in self.labels, then save their
        # indices
        if name not in self.groups:
            labels_indices = [self.labels.index(label) if type(label) is str
                              else label for label in labels]
            assert len(labels_indices) == len(set(labels_indices)) and \
                   all(type(index) is int for index in labels_indices)
            self.groups[name] = [labels_indices]
        else:
            raise RuntimeError

    @staticmethod
    def from_wrong_preds(labels, preds, reals, labels_count):
        """
        :param labels: label names
        :param preds: predicted
        :param reals: ground truth
        :param labels_count: total number of images per label
        :return: confusion matrix
        """
        conf = Confusion(labels)
        conf.update(preds, reals)
        conf.__complement_good_preds(labels_count)
        return conf

    @staticmethod
    def from_2d_list(list_of_lists, labels):
        conf = Confusion(labels)
        conf.matrix = np.array(list_of_lists)
        return conf

    def __complement_good_preds(self, labels_count):
        """Add entries from good predictions."""
        col_sums = self.matrix.sum(axis=0)
        good_preds = np.array(list(labels_count.values())) - col_sums
        self.matrix += np.diag(good_preds)

    def update(self, preds, reals):
        """Adds new results to existing confusion matrix."""
        # todo: update should pass parameters to existing Confusion c-tors and
        # todo: then perform __iadd__(self, other)
        for index, _ in enumerate(reals):
            self.matrix[preds[index], reals[index]] += 1

    def __str__(self):
        result = \
            f"acc:[{self.accuracy():.3f}] " \
            f"bin_acc:[{self.binary_accuracy():.3f}] " \
            f"lbl:[{' '.join(self.labels).strip()}] " \
            f"rec:[{' '.join([f'{a:.3f}' for a in self.recalls()])}] " \
            f"prec:[{' '.join([f'{a:.3f}' for a in self.precisions()])}] " \
            f"f1:[{' '.join([f'{a:.3f}' for a in self.f1scores()])}] "
        mcc = self.mcc()
        result += f"mcc: [{mcc:.3f}] " if mcc is not None else 'mcc: [NaN] '
        fnr = self.false_negative_rate()
        result += f"fnr: [{fnr:.3f}] " if fnr is not None else 'fnr: [NaN] '
        fpr = self.false_positive_rate()
        result += f"fpr: [{fpr:.3f}] " if fpr is not None else 'fpr: [NaN] '
        result += f"conf: {' '.join([str(row) for row in self.matrix])}"
        return result

    def accuracy(self):
        """Calculates accuracy."""
        trace = float(np.trace(self.matrix))
        total = float(self.matrix.sum())
        try:
            return trace / total
        except ZeroDivisionError:
            return 0

    def accuracies_per_class(self):
        return self.matrix.diagonal() / self.matrix.sum(axis=0)

    def mean_accuracy(self):
        accuracies_per_class = self.accuracies_per_class()
        return np.mean(accuracies_per_class), np.std(accuracies_per_class)

    def recalls(self):
        """
        Calculates recalls for each class. Zero-division check isn't
        needed unless there are labels with 0 examples associated.
        :return: vector of recalls where col_sum!=0 else 0
        """
        col_sums = self.matrix.sum(axis=0)
        diagonal = self.matrix.diagonal()
        return np.true_divide(diagonal, col_sums,
                              out=np.zeros_like(diagonal, dtype=float),
                              where=col_sums != 0)

    def precisions(self):
        """
        Calculates precision for each class, returns 0 if row_sum = 0.
        :return: vector of precisions
        """
        row_sums = self.matrix.sum(axis=1)
        diagonal = self.matrix.diagonal()
        return np.divide(diagonal, row_sums,
                         out=np.zeros_like(diagonal, dtype=float),
                         where=row_sums != 0)

    def f1scores(self):
        """
        Calculates f1 scores for each class. Returns 0 for classes which
        precision + recall = 0.
        :return: vector of f1-scores
        """
        precisions = self.precisions()
        recalls = self.recalls()
        nomin = 2 * precisions * recalls
        denom = precisions + recalls
        return np.divide(nomin, denom, out=np.zeros_like(nomin),
                         where=denom != 0)

    def binary_accuracy(self):
        """
        Calculates the accuracy of distinguishing the very first label
        from the rest.
        :return: binary accuracy
        """
        try:
            return (float(self.matrix[0, 0]) +
                    float(np.sum(self.matrix[1:, 1:]))) \
                   / float(np.sum(self.matrix))
        except ZeroDivisionError:
            return 0

    def false_negative_rate(self):
        if self.matrix.shape != (2, 2):
            return None
        m = self.matrix
        try:
            return float(m[1, 0]) / float(m[0, 0] + m[1, 0])
        except ZeroDivisionError:
            return None

    def false_positive_rate(self):
        if self.matrix.shape != (2, 2):
            return None
        m = self.matrix
        try:
            return float(m[0, 1]) / float(m[0, 1] + m[1, 1])
        except ZeroDivisionError:
            return None

    def mcc(self):
        if self.matrix.shape != (2, 2):
            return None
        m = self.matrix
        try:
            return float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) / \
                   float(((m[0, 0] + m[1, 0]) *
                          (m[0, 0] + m[0, 1]) *
                          (m[1, 1] + m[1, 0]) *
                          (m[1, 1] + m[0, 1])) ** 0.5)
        except ZeroDivisionError:
            return None

    def __add__(self, other):
        # we should probably do plenty of assertions in here
        result = copy.deepcopy(self)
        result.matrix = np.add(self.matrix, other.matrix)
        return result

    def __truediv__(self, other):
        """other should be a number"""
        result = copy.deepcopy(self)
        result.matrix = np.divide(self.matrix, other)
        return result

    def save(self, filepath):
        with open(filepath, "w") as file:
            file.write(json.dumps(self, indent=2))

    def to_json(self):
        return {'matrix': self.matrix.tolist(),
                'size': self.size,
                'labels': self.labels,
                'groups': self.groups
                }

    @staticmethod
    def from_json(data):
        return Confusion(data['labels'],
                         len(data['labels']),
                         np.array(data['matrix']))
