import logging
import numpy as np


class CustomProbaGenerator:
    def __init__(self, logger, choices, probas):
        self.__logger = logging.getLogger(__name__)
        self.__choices = choices
        self.__probas = probas
        self.__logger.info(f'Creating CustomProbaGenerator with {choices} and {probas}')
        self.__checkSettings()

    def next(self, size=1):
        return np.random.choice(self.__choices, size, p=self.__probas)

    def __checkSettings(self):
        if len(self.__choices) != len(self.__probas):
            raise ValueError('List of classes has different length than list of probabilities!')
        if len(self.__choices) == 0:
            raise ValueError(f'Got empty array of classes and probabilities!')
        probaSum = np.sum(self.__probas)
        if not np.isclose(probaSum, 1.0):
            raise ValueError(f'Sum of probabilities is not equal to 1.0 ({probaSum})!')
