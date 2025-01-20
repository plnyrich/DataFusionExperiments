from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import wandb


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


def repinit(cfg):
    if cfg['general']['use_wan_db'] is False:
        return

    run = wandb.init(
        project=f"plnyrich-{cfg['experiment']['tag']}-9901",
        tags=cfg['general']['wandb_tag'],
        config={
            'name': cfg['experiment']['name'],
            'tag': cfg['experiment']['tag'],
            'etag': cfg['general']['wandb_tag'],
            'classCount': len(cfg['generator']['classes']),
            'classifierCount': cfg['generator']['num_of_classifiers'],
            'classImbalance': [round(x, 2) for x in cfg['generator']['class_imbalance']],
            'steps': cfg['experiment']['steps'],
            'datasetSize': cfg['generator']['dataset_size'],
            'testSize': cfg['experiment']['test_size'],
            'randomState': cfg['experiment']['random_state'],
            'combinator': cfg['general']['combinator'],
        },
    )

    return run


def defineMetrics(useWanDb, run, methods):
    if not useWanDb:
        return

    #for m in methods:
    run.define_metric(
        f"rec-in",
        hidden=False
    )
    run.define_metric(
        f"acc",
        step_metric=f"rec-in",
        hidden=False
    )
    run.define_metric(
        f"pre",
        step_metric=f"rec-in",
        hidden=False
    )
    run.define_metric(
        f"rec",
        step_metric=f"rec-in",
        hidden=False
    )
    run.define_metric(
        f"f1",
        step_metric=f"rec-in",
        hidden=False
    )



def report(cfg, stats, step):
    if cfg['general']['use_wan_db'] is False:
        return

    for m in stats.methods():
        data = []
        d = stats.stats(m)
        for i in range(len(d['x'])):
            data.append([float(d['x'][i]), float(d['y'][i])])

        wandb.log({
            f'graphs/graph_{m}': linePlot,
            f'eval/report_{m}': table
        })


def repclose(run):
    if run is None:
        return

    run.finish()
