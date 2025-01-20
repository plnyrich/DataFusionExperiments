#!/bin/bash

echo "Experiment stared: $(date)" > experiment.log

echo "BQ1 started: $(date)" >> experiment.log
source ./runners/run_bq1.sh

echo "BQ2 started: $(date)" >> experiment.log
source ./runners/run_bq2.sh

echo "NQ3 started: $(date)" >> experiment.log
source ./runners/run_nq3.sh

echo "Experiment ended: $(date)" >> experiment.log
