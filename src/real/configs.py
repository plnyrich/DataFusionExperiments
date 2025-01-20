def classesForDataset(dataset):
    if dataset == 'quic40_asn' or dataset == 'quic40':
        return list(range(40))
    elif dataset == 'tor':
        return [0, 1]
    else:
        raise ValueError(f'Dataset {dataset} is not known!')


def clfsForDataset(dataset):
    if dataset == 'quic40':
        return 4
    elif dataset == 'quic40_asn':
        return 5
    elif dataset == 'tor':
        return 3
    else:
        raise ValueError(f'Dataset {dataset} is not known!')


def labelForDataset(dataset):
    if dataset == 'quic40' or dataset == 'quic40_asn':
        return 'LABEL'
    elif dataset == 'doh':
        return 'ENC_LABEL'
    elif dataset == 'tor':
        return 'ENC_LABEL'
    else:
        raise ValueError(f'Dataset {dataset} is not known!')


def createCfg(dataset, randomState=42):
    return {
        'experiment_name': 'verif',
        'dataset_folder': '/opt/dev',
        'dataset_name_train': f'{dataset}_train',
        'dataset_name_test': f'{dataset}_test',
        'output_file': f'realResults_{dataset}.txt',
        'random_state': randomState,
        'round_precision': 4,
        'classes': classesForDataset(dataset),
        'num_of_classifiers': clfsForDataset(dataset),
        'label_field': labelForDataset(dataset),
        'field_prefix': 'CLF'
    }
