import datetime
import logging

from src.generator.dataGen import DataGenerator


def runGenerator(cfg):
    logger = logging.getLogger(__name__)
    logger.info('Running data generator...')

    try:
        dg = DataGenerator(cfg, logger)
        dataset = dg.sample(cfg['dataset_size'])

        datasetFile = f"{cfg['dataset_folder']}/{cfg['dataset_name']}.csv"
        dataset.to_csv(datasetFile, index=False)
    except Exception as e:
        logger.critical(f'{e}')
        print(f'runGenerator(): {e}')

    logger.info('Data generator finished')
