import math

from src.common.common import fixMetric
from sklearn.metrics import recall_score


class BasicRecallCombinator:
    def __init__(self, defs):
        self.__classes = defs['classes']
        self.__fields = defs['fields']
        self.__label = defs['label']
        self.__weights = []
        self.classConstants = []

    def name(self):
        return 'basic-recall'

    def shortName(self):
        return 'bas-rec'

    def field(self):
        return 'RCB_LABEL'

    def train(self, df):
        self.__weights = []

        for clfField in self.__fields:
            recallsForAllClasses = recall_score(df[self.__label], df[clfField], labels=self.__classes, average=None, zero_division=0.0)
            self.__weights.append([self.__calculateWeight(x) for x in recallsForAllClasses])

    def __calculateWeight(self, recall):
        recall = self.__fix(recall)
        return math.log2(recall/(1-recall))

    def combine(self, row):
        score = [0.0] * len(self.__classes)

        for clfIdx, field in enumerate(self.__fields):
            score[row[field]] += self.__weights[clfIdx][row[field]]

        argmax = max(enumerate(score), key=lambda x: x[1])
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
