import argparse
import numpy as np
import os


ROOT_FOLDER = 'experiments'
CLF_COUNTS = [3, 5, 7]
CLS_COUNTS = [3, 5, 7]
DATASET_SIZE = 100_000
REC_CLF = 1.0
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


def findClsSimStart(data):
    for idx, line in enumerate(data):
        if line.find('%CLS_SIMILARITIES') != -1:
            return idx
    return -1


def putRecalls(lines, clfIdx, clfCount, recall):
    for idx, line in enumerate(lines):
        if line.find(f'%RECALLS_{clfIdx}') != -1:
            recStr = ', '.join([f"{x}" for x in recall])
            lines[idx] = line.replace(f'%RECALLS_{clfIdx}', f"{clfIdx}: [{recStr}]")
            return lines
    return lines


def clsSimilaritiesForClass(clsIdx, clsCount):
    s = ['1'] * clsCount
    s[clsIdx] = '0'
    return ', '.join(s)


def putClsSims(lines, clfIdx, clsCount):
    for idx, line in enumerate(lines):
        if line.find(f'%CLS_SIMILARITIES_{clfIdx}') != -1:
            spaceCnt = line.find('%')
            starter = ' ' * spaceCnt
            lines.insert(idx + 1, f'{starter}{clfIdx}:\n')
            starter = ' ' * (spaceCnt + 2)
            for clsIdx in range(clsCount):
                lines.insert(idx + clsIdx + 2, f'{starter}{clsIdx}: [{clsSimilaritiesForClass(clsIdx, clsCount)}]\n')
            del lines[idx]
            return lines
    return lines


def fint(i, n):
    return str(i).rjust(n, '0')


def createConfig(confId, cix, combinatorName, rs, clsCount, clfCount, clsImbalance, recalls):
    confName = f"{fint(confId, 3)}_{clfCount}_{clsCount}_{rs}_{cix}"
    datasetName = f"{DATASET_NAME}_{fint(confId, 3)}_{clfCount}_{clsCount}_{rs}_{cix}"
    resultsFile = f"{RESULTS}/{fint(confId, 3)}_{clfCount}_{clsCount}_{rs}_{cix}.txt"
    experimentName = f"{EXPERIMENT_NAME}-{fint(confId, 3)}-{clfCount}-{clsCount}-RS{rs}-C{cix}"
    #wandbTag = f"{fint(confId, 3)}_{clfCount}_{clsCount}"

    wandbTags = []
    wandbTags.append(combinatorName)
    wandbTags.append(f"clf_{clfCount}")
    wandbTags.append(f"cls_{clsCount}")

    with open('./scripts/template_nq3.yaml', 'rt') as src:
        lines = src.readlines()

    lines = expandVariable(lines, 'EXPERIMENT_NAME', experimentName)
    lines = expandVariable(lines, 'TAG', TAG)
    lines = expandVariable(lines, 'WANDB_TAGS', wandbTags)
    lines = expandVariable(lines, 'COMBINATOR', combinatorName)
    lines = expandVariable(lines, 'DATASET_FOLDER', DATASET_FOLDER)
    lines = expandVariable(lines, 'OUTFILE', resultsFile)
    lines = expandVariable(lines, 'DATASET_NAME', datasetName)
    lines = expandVariable(lines, 'DATASET_SIZE', DATASET_SIZE)
    lines = expandVariable(lines, 'CLF_COUNT', clfCount)
    lines = expandVariable(lines, 'RANDOM_STATE', str(rs))
    lines = expandVariable(lines, 'CLASSES', ', '.join([f"{x}" for x in range(clsCount)]))
    lines = expandVariable(lines, 'CLS_IMBALANCE', ', '.join([f"{x}" for x in clsImbalance]))
    lines = expandVariable(lines, 'STEPS', len(recalls))

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

    clsSimIdx = findClsSimStart(lines)
    for i in range(clfCount):
        lines.insert(clsSimIdx + i + 1, f"{lines[clsSimIdx].rstrip()}_{i}\n")
    del lines[clsSimIdx]

    for i in range(clfCount):
        lines = putClsSims(lines, i, clsCount)

    with open(f'{CONFS}/{confName}.yaml', 'wt') as dst:
        dst.write(''.join(lines))


def fixRecalls(arr):
    for idx, e in enumerate(arr):
        if e > 1.0:
            arr[idx] = 1
    return arr


def main():
    os.makedirs(CONFS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    confId = 0
    for clsCount in CLS_COUNTS:
        for clfCount in CLF_COUNTS:
            recallSteps = []

            recalls = []
            for _ in range(clfCount):
                recalls.append([REC_CLF] * clsCount)

            recalls = np.array(recalls)
            clsImbalance = np.array([1]*clsCount) / clsCount

            while np.any(recalls >= REC_END):
                currRecs = recalls.copy()
                recallSteps.append(currRecs)
                recalls = np.round(recalls - REC_STEP, 4)

            for rs in RANDOM_STATES:
                for cix, combinatorName in enumerate(COMBINATORS):
                    createConfig(
                        confId,
                        cix,
                        combinatorName,
                        rs,
                        clsCount,
                        clfCount,
                        clsImbalance,
                        recallSteps
                    )
            confId += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NQ3 Conf Generator'
    )

    global TAG
    TAG = 'nq3'

    global EXPERIMENT_NAME
    EXPERIMENT_NAME = 'E-N3'

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
