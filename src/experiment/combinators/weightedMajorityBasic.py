import math

from src.common.common import fixMetric
from sklearn.metrics import accuracy_score


class BasicWeightedMajorityCombinator:
    def __init__(self, defs):
        self.__classes = defs['classes']
        self.__fields = defs['fields']
        self.__label = defs['label']
        self.__weights = []

    def name(self):
        return 'basic-weighted-majority'

    def shortName(self):
        return 'bas-wmj'

    def field(self):
        return 'WMB_LABEL'

    def train(self, df):
        # per classifier weights
        self.__weights = []
        accuracies = []

        for clfField in self.__fields:
            accuracy = self.__fix(accuracy_score(df[self.__label], df[clfField]))
            self.__weights.append(math.log2(accuracy/(1.0-accuracy)))
            accuracies.append(accuracy)

        #print(accuracies)
        #print(self.__weights)
        #print()

    def combine(self, row):
        #print(row)

        score = [0.0] * len(self.__classes)
        for clfIdx, field in enumerate(self.__fields):
            score[row[field]] += self.__weights[clfIdx]

        argmax = max(enumerate(score), key=lambda x: x[1])
        #print(f'{score} , result : {argmax[0]}')
        #print()
        return argmax[0]

    @staticmethod
    def __fix(metric):
        eps = 0.001
        if metric == 0.0:
            return metric + eps
        elif metric == 1.0:
            return metric - eps
        else:
            return metric
