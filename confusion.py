import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Confusion:
    """
    Class for storing confusion matrix and calculating its statistics.
    """
    labels: [str]
    size: int
    matrix: np.array

    @staticmethod
    def empty(labels):
        """
        Creates confusion mat. of size n x n, where n is the len(labels)
        """
        size = len(labels)
        return Confusion(labels, size, np.zeros((size, size)))

    @staticmethod
    def from_wrong_preds(labels, preds, reals, labels_count):
        """
        :param labels: label names
        :param preds: predicted
        :param reals: ground truth
        :param labels_count: total number of images per label
        :return: confusion matrix
        """
        conf = Confusion.empty(labels)
        conf.update(preds, reals)
        conf.__complement_good_preds(labels_count)
        return conf

    def __complement_good_preds(self, labels_count):
        """Add entries from good predictions."""
        col_sums = self.matrix.sum(axis=0)
        good_preds = np.array(list(labels_count.values())) - col_sums
        self.matrix += np.diag(good_preds)

    def update(self, preds, reals):
        """Adds new results to existing confusion matrix."""
        for index, _ in enumerate(reals):
            self.matrix[preds[index], reals[index]] += 1

    def __str__(self):
        return f"acc:[{self.accuracy():.3f}] bin_acc:[{self.binary_accuracy():.3f}] " \
               f"lbl:[{' '.join(self.labels).strip()}]" \
               f"rec:[{' '.join([f'{a:.3f}' for a in self.recalls()]).strip()}] " \
               f"prec:[{' '.join([f'{a:.3f}' for a in self.precisions()]).strip()}] " \
               f"f1:[{' '.join([f'{a:.3f}' for a in self.f1scores()])}]"

    def accuracy(self):
        """Calculates accuracy."""
        trace = float(np.trace(self.matrix))
        total = float(self.matrix.sum())
        try:
            return trace/total
        except ZeroDivisionError:
            return 0

    def recalls(self):
        """
        Calculates recalls for each class. Zero-division check isn't
        needed unless there are labels with 0 examples associated.
        :return: vector of recalls where col_sum!=0 else 0
        """
        col_sums = self.matrix.sum(axis=0)
        diagonal = self.matrix.diagonal()
        return np.divide(diagonal, col_sums,
                         out=np.zeros_like(diagonal),
                         where=col_sums != 0)

    def precisions(self):
        """
        Calculates precision for each class, returns 0 if row_sum = 0.
        :return: vector of precisions
        """
        row_sums = self.matrix.sum(axis=1)
        diagonal = self.matrix.diagonal()
        return np.divide(diagonal, row_sums,
                         out=np.zeros_like(diagonal),
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
            return (float(self.matrix[0, 0]) + float(np.sum(self.matrix[1:, 1:])))\
                   / float(np.sum(self.matrix))
        except ZeroDivisionError:
            return 0

    def save(self, filepath):
        with open(filepath, "w") as file:
            file.write(json.dumps(self.to_json(), indent=2))

    def to_json(self):
        return {'matrix': self.matrix.tolist(),
                'size': self.size,
                'labels': self.labels,
                'stats': {'accuracy': self.accuracy(),
                          'recalls': self.recalls().tolist(),
                          'precisions': self.precisions().tolist(),
                          'f1scores': self.f1scores().tolist(),
                          'bin_acc': self.binary_accuracy()
                          }
                }

    @staticmethod
    def from_json(data):
        return Confusion(data['labels'],
                         len(data['labels']),
                         np.array(data['matrix']))
