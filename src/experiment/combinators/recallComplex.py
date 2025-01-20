import math
import copy

from src.common.common import fixMetric
from sklearn.metrics import recall_score


class ComplexRecallCombinator:
    def __init__(self, defs):
        self.__classes = defs['classes']
        self.__fields = defs['fields']
        self.__label = defs['label']
        self.__weights = []
        self.__classConstants = []
        self.__var = 0

    def name(self):
        return 'complex-recall'

    def shortName(self):
        return 'cmx-rec'

    def field(self):
        return 'RCC_LABEL'

    def train(self, df):
        self.__weights = []
        self.__classConstants = []

        for trueClass in self.__classes:
            classProbability = self.__fix(len(df[df[self.__label] == trueClass])/len(df))
            self.__classConstants.append(math.log2(classProbability))

        for clfField in self.__fields:
            recallsForAllClasses = recall_score(df[self.__label], df[clfField], labels=self.__classes, average=None, zero_division=0.0)
            self.__weights.append([self.__calculateWeight(x) for x in recallsForAllClasses])
            for classIdx, classRecall in enumerate(recallsForAllClasses):
                self.__classConstants[classIdx] += math.log2(self.__fix(1 - classRecall))

        if len(self.__classes) > 1:
            self.__var = math.log2(len(self.__classes) - 1)

    def __calculateWeight(self, recall):
        recall = self.__fix(recall)
        return math.log2(recall/(1-recall))

    def combine(self, row):
        score = copy.deepcopy(self.__classConstants)

        for clfIdx, field in enumerate(self.__fields):
            score[row[field]] += self.__weights[clfIdx][row[field]]
            score[row[field]] += self.__var

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
