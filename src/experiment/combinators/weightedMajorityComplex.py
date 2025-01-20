import math
import copy

from src.common.common import fixMetric
from sklearn.metrics import accuracy_score


class ComplexWeightedMajorityCombinator:
    def __init__(self, defs):
        self.__classes = defs['classes']
        self.__fields = defs['fields']
        self.__label = defs['label']
        self.__weights = []
        self.__classConstants = []
        self.__var = 0

    def name(self):
        return 'complex-weighted-majority'

    def shortName(self):
        return 'cmx-wmj'

    def field(self):
        return 'WMC_LABEL'

    def train(self, df):
        # per classifier weights
        self.__weights = []
        # class constants
        self.__classConstants = []
        accuracies = []

        for trueClass in self.__classes:
            classConstant = self.__fix(len(df[df[self.__label] == trueClass])/len(df))
            self.__classConstants.append(math.log2(classConstant))

        for clfField in self.__fields:
            accuracy = self.__fix(accuracy_score(df[self.__label], df[clfField]))
            self.__weights.append(math.log2(accuracy/(1.0-accuracy)))
            accuracies.append(accuracy)

        if len(self.__classes) > 1:
            self.__var = math.log2(len(self.__classes) - 1)

    def combine(self, row):
        #print(row)

        score = copy.deepcopy(self.__classConstants)
        for clfIdx, field in enumerate(self.__fields):
            score[row[field]] += self.__weights[clfIdx]
            score[row[field]] += self.__var

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
