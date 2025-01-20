import argparse
import numpy as np
import os


ROOT_FOLDER = 'experiments'
CLF_COUNTS = [3, 5, 7]
DATASET_SIZE = 100_000
CLASS_IMBALANCES = [
    #[0.5, 0.5],
    #[0.9, 0.1],
    [0.99, 0.01]
]
CLF_REC_START = 1.0
REC_STEP = 0.025
REC_END = 0.48

COMBINATORS = [
    'maj',
    'bas-wmj',
    'cmx-wmj',
    'bas-rec',
    'cmx-rec',
    'naive-bayes',
    'bks',
    'dec-tree',
    'log-reg',
    'oracle'
]


def expandVariable(data, variableName, variableValue):
    for idx, line in enumerate(data):
        data[idx] = line.replace(f'${variableName}', str(variableValue))
    return data


def findRecallsStart(data):
    for idx, line in enumerate(data):
        if line.find('%RECALLS') != -1:
            return idx
    return -1


def findRecallsNStart(data, n):
    for idx, line in enumerate(data):
        if line.find(f'%RECALLS_{n}') != -1:
            return idx
    return -1


def putRecalls(lines, clfIdx, clfCount, recall):
    for idx, line in enumerate(lines):
        if line.find(f'%RECALLS_{clfIdx}') != -1:
            recStr = ', '.join([f"{x}" for x in recall])
            lines[idx] = line.replace(f'%RECALLS_{clfIdx}', f"{clfIdx}: [{recStr}]")
            return lines
    return lines


def imbToIndex(clsImbl):
    for i, x in enumerate([[0.5, 0.5],[0.9, 0.1],[0.99, 0.01]]):
        if x == clsImbl:
            return i
    raise ValueError(clsImbl)


def imbToString(clsImbl):
    texts = [
        '50-50',
        '90-10',
        '99-1'
    ]

    for i, x in enumerate([[0.5, 0.5],[0.9, 0.1],[0.99, 0.01]]):
        if x == clsImbl:
            return texts[i]
    raise ValueError(clsImbl)


def fint(i, n):
    return str(i).rjust(n, '0')


def createConfig(confId, cix, combinatorName, rs, clfCount, clsImbalance, recalls):
    confName = f"{fint(confId, 3)}_{clfCount}_{imbToIndex(clsImbalance)}_{rs}_{cix}"
    datasetName = f"{DATASET_NAME}_{fint(confId, 3)}_{clfCount}_{imbToIndex(clsImbalance)}_{rs}_{cix}"
    resultsFile = f"{RESULTS}/{fint(confId, 3)}_{clfCount}_{imbToIndex(clsImbalance)}_{rs}_{cix}.txt"
    experimentName = f"{EXPERIMENT_NAME}-{fint(confId, 3)}-{clfCount}-{imbToIndex(clsImbalance)}-RS{rs}-C{cix}"
    #wandbTag = f"{fint(confId, 3)}_{clfCount}_{imbToIndex(clsImbalance)}"

    wandbTags = []
    wandbTags.append(combinatorName)
    wandbTags.append(f"clf_{clfCount}")
    wandbTags.append(f"imb_{imbToString(clsImbalance)}")

    with open('./scripts/template.yaml', 'rt') as src:
        lines = src.readlines()

    lines = expandVariable(lines, 'EXPERIMENT_NAME', experimentName)
    lines = expandVariable(lines, 'TAG', TAG)
    lines = expandVariable(lines, 'COMBINATOR', combinatorName)
    lines = expandVariable(lines, 'WANDB_TAGS', wandbTags)
    lines = expandVariable(lines, 'DATASET_FOLDER', DATASET_FOLDER)
    lines = expandVariable(lines, 'OUTFILE', resultsFile)
    lines = expandVariable(lines, 'DATASET_NAME', datasetName)
    lines = expandVariable(lines, 'DATASET_SIZE', DATASET_SIZE)
    lines = expandVariable(lines, 'CLF_COUNT', clfCount)
    lines = expandVariable(lines, 'RANDOM_STATE', str(rs))
    lines = expandVariable(lines, 'CLS_IMBALANCE', ', '.join([f"{x}" for x in clsImbalance]))
    lines = expandVariable(lines, 'STEPS', len(recalls))
    lines = expandVariable(lines, 'USE_ALL_RECALLS', 'True')

    recallIdx = findRecallsStart(lines)
    for i in range(len(recalls)):
        lines.insert(recallIdx + i + 1, f"{lines[recallIdx].rstrip()}_{i}\n")
    del lines[recallIdx]

    for i in range(len(recalls)):
        start = findRecallsNStart(lines, i)
        lines.insert(start + 1, f"  recalls_for_{i}:\n")
        for j in range(clfCount):
            lines.insert(start + j + 2, f"    $recalls_step_{fint(i, 3)}_{fint(j, 3)}\n")
        del lines[start]

    for i in range(len(recalls)):
        for j in range(clfCount):
            arr = ', '.join([str(x) for x in recalls[i][j]])
            lines = expandVariable(lines, f'recalls_step_{fint(i, 3)}_{fint(j, 3)}', f'{j}: [{arr}]')

    with open(f'{CONFS}/{confName}.yaml', 'wt') as dst:
        dst.write(''.join(lines))


def fixRecalls(arr):
    for idx, e in enumerate(arr):
        if e > 1.0:
            arr[idx] = 1
    return np.round(arr, 4)


def main():
    os.makedirs(CONFS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    confId = 0
    for clfCount in CLF_COUNTS:
        for clsImbalance in CLASS_IMBALANCES:
            for rs in RANDOM_STATES:
                np.random.seed(rs)

                currentRecall = CLF_REC_START
                recallSteps = []
                while currentRecall >= REC_END:
                    recalls = []

                    # put recalls for every clf
                    for clfIdx in range(clfCount):
                        recs = [currentRecall, currentRecall]
                        recalls.append(fixRecalls(recs))

                    recallSteps.append(recalls)
                    currentRecall -= REC_STEP

                    for cix, combinatorName in enumerate(COMBINATORS):
                        createConfig(
                            confId,
                            cix,
                            combinatorName,
                            rs,
                            clfCount,
                            clsImbalance,
                            recallSteps
                        )
            confId += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='BQ2 Conf Generator'
    )

    global TAG
    TAG = 'bq2'

    global EXPERIMENT_NAME
    EXPERIMENT_NAME = 'E-B2'

    global RANDOM_STATES
    RANDOM_STATES = [137, 359, 557, 821, 941]

    global DATASET_NAME
    DATASET_NAME = f'{TAG}'

    global DATASET_FOLDER
    DATASET_FOLDER = f'{ROOT_FOLDER}/datasets/{TAG}'

    global CONFS
    CONFS = f'{ROOT_FOLDER}/confs/{TAG}'

    global RESULTS
    RESULTS = f'{ROOT_FOLDER}/results/{TAG}'

    main()
