#!/bin/bash

source venv/bin/activate

ROOT_FOLDER="experiments"
TAG="nq3"

# Clean
rm -rf ${ROOT_FOLDER}/confs/${TAG}
rm -rf ${ROOT_FOLDER}/datasets/${TAG}
rm -rf ${ROOT_FOLDER}/results/${TAG}
rm -rf "logs/log_E-N3"*

mkdir -p ${ROOT_FOLDER}/datasets/${TAG}
mkdir -p ${ROOT_FOLDER}/results/${TAG}

python scripts/generate_confs_nq3.py

CONF_FOLDER=${ROOT_FOLDER}/confs/${TAG}
CONFS_TO_RUN=$(ls -1 $CONF_FOLDER/*.yaml)
NUM_OF_CONFS_TO_RUN=$(ls -1 $CONF_FOLDER/*.yaml | wc -l)

COUNTER=0
for CONF_TO_RUN in $CONFS_TO_RUN
do
	CONF_TO_RUN=$(echo $CONF_TO_RUN | sed 's/.yaml//g' | cut -d'/' -f4)

	python main.py --config-dir $CONF_FOLDER --config-name $CONF_TO_RUN &
	#echo "python main.py --config-dir $CONF_FOLDER --config-name $CONF_TO_RUN &"

	COUNTER=$(expr $COUNTER + 1)

	if [ "$(( $COUNTER % 10 ))" -eq 0 ]; then
		sleep 1
		wait $(jobs -p)
		echo "[$(date)] Status: ${COUNTER} / ${NUM_OF_CONFS_TO_RUN}"
	fi
done

sleep 10
