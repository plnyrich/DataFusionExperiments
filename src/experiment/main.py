import datetime
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.experiment.evals import eval_results

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


def selectCombinator(combinatorName, combinators):
    for combinator in combinators:
        if combinator.shortName() == combinatorName:
            return combinator


def runExperiment(cfg, wandb_run, step):
    logger = logging.getLogger(__name__)
    logger.info('Running experiment...')

    dataset = f"{cfg['dataset_folder']}/{cfg['dataset_name']}.csv"
    logger.info(f'Using {dataset}')

    df = pd.read_csv(dataset)
    df_train, df_test = train_test_split(df, test_size=cfg['test_size'], random_state=cfg['random_state'])

    logger.info(f'Train size : {len(df_train)}')
    logger.info(f'Test size : {len(df_test)}')

    defs = createDefs(cfg)

    try:
        combinators = combinatorDefs(defs)

        resultsFile = open(cfg['output_file'], 'at')
        combinator = selectCombinator(cfg['combinator'], combinators)

        logger.info(f'Running {combinator.name()}...')
        combinator.train(df_train)
        logger.info(f'Train done')
        df_test[combinator.field()] = df_test.apply(combinator.combine, axis=1)
        acc, rec, pre, f1 = eval_results(
            df_test[defs['label']],
            df_test[combinator.field()],
            cfg['classes'],
            combinator.name(),
            defs["round_precision"],
            resultsFile
        )

        wandb_metrics = {}
        if cfg['use_all_recalls']:
            s = 0
            cnt = 0
            for k, v in cfg['recalls'].items():
                s += np.sum(v)
                cnt += len(v)
            avg = round(s / cnt, 4)
            wandb_metrics[f'rec-in'] = round(s / cnt, 4)
        else:
            wandb_metrics[f'rec-in'] = np.average(cfg['recalls'][0])
        wandb_metrics[f'acc'] = acc
        wandb_metrics[f'rec'] = rec
        wandb_metrics[f'pre'] = pre
        wandb_metrics[f'f1'] = f1

        if cfg['use_wan_db']:
            wandb_run.log(wandb_metrics, step=step, commit=True)

        resultsFile.close()
        logger.info('Experiment finished')
    except Exception as e:
        logger.error(f'Experiment: {e}')
