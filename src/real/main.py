import datetime
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.experiment.evals import eval_results
from src.real.configs import createCfg

from src.experiment.combinators.majority import MajorityCombinator
from src.experiment.combinators.weightedMajorityBasic import BasicWeightedMajorityCombinator
from src.experiment.combinators.weightedMajorityComplex import ComplexWeightedMajorityCombinator
from src.experiment.combinators.recallBasic import BasicRecallCombinator
from src.experiment.combinators.recallComplex import ComplexRecallCombinator
from src.experiment.combinators.naiveBayes import NaiveBayesCombinator
from src.experiment.combinators.bks import BKSCombinator
from src.experiment.combinators.decisionTree import DecisionTreeCombinator
from src.experiment.combinators.logRegression import LogisticRegressionCombinator
from src.experiment.combinators.oracle import OracleCombinator


def createDefs(cfg):
    defs = dict()
    defs['classes'] = cfg['classes']
    defs['num_of_classifiers'] = cfg['num_of_classifiers']
    defs['fields'] = [f'{cfg["field_prefix"]}_{i}' for i in range(cfg['num_of_classifiers'])]
    defs['label'] = cfg['label_field']
    defs['round_precision'] = cfg['round_precision']
    return defs


def combinatorDefs(defs):
    return [
        MajorityCombinator(defs),
        #BasicWeightedMajorityCombinator(defs),
        ComplexWeightedMajorityCombinator(defs),
        #BasicRecallCombinator(defs),
        ComplexRecallCombinator(defs),
        NaiveBayesCombinator(defs),
        BKSCombinator(defs),
        DecisionTreeCombinator(defs),
        LogisticRegressionCombinator(defs),
        OracleCombinator(defs)
    ]


def runExperiment(dataset):
    cfg = createCfg(dataset)
    defs = createDefs(cfg)

    dataset_train = f"{cfg['dataset_folder']}/{cfg['dataset_name_train']}.csv"
    df_train = pd.read_csv(dataset_train)

    dataset_test = f"{cfg['dataset_folder']}/{cfg['dataset_name_test']}.csv"
    df_test = pd.read_csv(dataset_test)

    print(f'TRAIN SIZE {len(df_train)}')
    print(f'TEST SIZE  {len(df_test)}')

    try:
        resultsFile = open(cfg['output_file'], 'wt')
        for combinator in combinatorDefs(defs):
            print(f'Running {combinator.name()}...')
            combinator.train(df_train)
            print(f'Train done')
            df_test[combinator.field()] = df_test.apply(combinator.combine, axis=1)
            acc, rec, pre, f1 = eval_results(
                df_test[defs['label']],
                df_test[combinator.field()],
                cfg['classes'],
                combinator.name(),
                defs["round_precision"],
                resultsFile
            )

        resultsFile.close()
        #df_test.to_csv(f"{cfg['dataset_folder']}/{cfg['dataset_name_test']}_processed.csv", index=False)
        print('Experiment finished')
    except Exception as e:
        print(f'Experiment: {e}')
