import datetime
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from src.generator.main import runGenerator
from src.experiment.main import runExperiment
from src.reporter.main import repinit, report, repclose
from src.reporter.stats import ExpStats

from matplotlib import pyplot as plt


def createGeneratorConfig(cfg, step, rs):
    return {
        'dataset_size': cfg['generator']['dataset_size'],
        'dataset_folder': cfg['general']['dataset_folder'],
        'dataset_name': cfg['general']['dataset_name'],
        'rng_seed': rs,
        'label_field': cfg['general']['ground_truth_field_name'],
        'field_prefix': cfg['general']['classifier_field_prefix'],
        'num_of_classifiers': cfg['generator']['num_of_classifiers'],
        'recalls': cfg['generator'][f'recalls_for_{step}'],
        'classes': cfg['generator']['classes'],
        'class_imbalance': cfg['generator']['class_imbalance'],
        'class_similarities': cfg['generator']['class_similarities']
    }


def createExperimentConfig(cfg, step, rs):
    return {
        'combinator': cfg['general']['combinator'],
        'experiment_name': cfg['experiment']['name'],
        'dataset_folder': cfg['general']['dataset_folder'],
        'dataset_name': cfg['general']['dataset_name'],
        'output_file': cfg['general']['output_file'],
        'random_state': rs,
        'round_precision': cfg['general']['round_precision'],
        'test_size': cfg['experiment']['test_size'],
        'classes': cfg['generator']['classes'],
        'num_of_classifiers': cfg['generator']['num_of_classifiers'],
        'label_field': cfg['general']['ground_truth_field_name'],
        'field_prefix': cfg['general']['classifier_field_prefix'],
        'recalls': cfg['generator'][f'recalls_for_{step}'],
        'use_all_recalls': cfg['general']['use_all_recalls'],
        'use_wan_db': cfg['general']['use_wan_db']
    }


def methodToLabel(method):
    if method == 'majority':
        return 'maj'
    elif method == 'weighted-majority':
        return 'wmj'
    elif method == 'recall':
        return 'rec'
    elif method == 'naive-bayes':
        return 'nbs'
    elif method == 'bks':
        return 'bks'
    elif method == 'decision-tree':
        return 'dct'
    elif method == 'log-regression':
        return 'log'
    elif method == 'dst':
        return 'dst'
    elif method == 'orc':
        return 'orc'


def methodToColor(method):
    if method == 'majority':
        return 'tab:blue'
    elif method == 'weighted-majority':
        return 'tab:orange'
    elif method == 'recall':
        return 'tab:green'
    elif method == 'naive-bayes':
        return 'tab:red'
    elif method == 'bks':
        return 'tab:purple'
    elif method == 'decision-tree':
        return 'tab:brown'
    elif method == 'log-regression':
        return 'tab:pink'
    elif method == 'dst':
        return 'tab:gray'
    elif method == 'orc':
        return 'tab:olive'


@hydra.main(version_base=None, config_path="confs", config_name="basic")
def app(cfg : DictConfig) -> None:
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logFileName = f"log_{cfg['experiment']['name']}_{ts}.log"
    logging.basicConfig(
        filemode='w',
        filename=f'logs/{logFileName}',
        force=True,
        format='[%(asctime)s][%(levelname)-8s] %(message)s',
        level=logging.INFO
    )

    run = repinit(cfg)

    rs = cfg['experiment']['random_state']
    for step in range(cfg['experiment']['steps']):
        runGenerator(createGeneratorConfig(cfg, step, rs))
        runExperiment(createExperimentConfig(cfg, step, rs), run, step)

    repclose(run)

if __name__ == '__main__':
    app()
