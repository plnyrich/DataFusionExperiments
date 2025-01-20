import logging
import numpy as np
import pandas as pd

from src.generator.customProbaGen import CustomProbaGenerator


class DataGenerator:
    def __init__(self, config, logger):
        self.__config = config
        self.__logger = logging.getLogger(__name__)

        self.__seedRng()

        # Ground Truth Generator
        self.__gtGenerator = CustomProbaGenerator(self.__logger, list(range(len(self.__config['classes']))), self.__config['class_imbalance'])
        # Generators of Classifiers' Decisions
        self.__clfDecisionGenerators = []
        for clfIdx in range(self.__config['num_of_classifiers']):
            decisionGenerators = []
            for classRecall in self.__config['recalls'][clfIdx]:
                decisionGenerators.append(
                    CustomProbaGenerator(
                        self.__logger,
                        [True, False],
                        [classRecall, 1.0 - classRecall]
                    )
                )
            self.__clfDecisionGenerators.append(decisionGenerators)
        # Generators of Classifiers' Outputs
        self.__clfOutputGenerators = []
        for clfIdx in range(self.__config['num_of_classifiers']):
            outputGenerators = []
            for classIdx in range(len(self.__config['classes'])):
                outputGenerators.append(
                    CustomProbaGenerator(
                        self.__logger,
                        list(range(len(self.__config['classes']))),
                        self.__config['class_similarities'][clfIdx][classIdx] / np.sum(self.__config['class_similarities'][clfIdx][classIdx])
                    )
                )
            self.__clfOutputGenerators.append(outputGenerators)

    def __seedRng(self):
        self.__logger.info(f'Seeding RNG with {self.__config["rng_seed"]}')
        np.random.seed(self.__config['rng_seed'])

    def sample(self, size):
        data = self.__createEmptyDf()
        groundTruths = self.__gtGenerator.next(size)

        for groundTruth in groundTruths:
            data[self.__config['label_field']].append(groundTruth)
            clfOutputs = self.__generateClfOutputs(groundTruth)
            for clfIdx in range(self.__config['num_of_classifiers']):
                data[f"{self.__config['field_prefix']}_{clfIdx}"].append(clfOutputs[clfIdx])

        return pd.DataFrame.from_dict(data)

    def __createEmptyDf(self):
        data = dict()
        data[self.__config['label_field']] = []
        for clf in range(self.__config['num_of_classifiers']):
            data[f"{self.__config['field_prefix']}_{clf}"] = []
        return data

    def __generateClfOutputs(self, groundTruth):
        outputs = []
        for clfIdx in range(self.__config['num_of_classifiers']):
            outputs.append(self.__generateClfOutput(clfIdx, groundTruth))
        return outputs

    def __generateClfOutput(self, clfIdx, groundTruth):
        generateCorrectOutput = self.__clfDecisionGenerators[clfIdx][groundTruth].next()
        if generateCorrectOutput:
            # generate correct label
            return groundTruth
        else:
            # generate incorrect label
            return self.__clfOutputGenerators[clfIdx][groundTruth].next()[0]
