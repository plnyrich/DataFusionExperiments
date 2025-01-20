# Data Fusion Experiments

## Preparation
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running synthetic experiments
Experiments on synthetic data use [hydra.cc](https://hydra.cc). Template configs are available in [template.yaml](scripts/template.yaml) for BQ1 and BQ2 scenarios, and in [template_nq3.yaml](scripts/template_nq3.yaml) for NQ3.

You edit scenario parameters and generate configs by [generate_confs_bq1.yaml](scripts/generate_confs_bq1.py) (for BQ1), [generate_confs_bq2.yaml](scripts/generate_confs_bq2.py) for BQ2, and [generate_confs_nq3.yaml](scripts/generate_confs_nq3.py) (for NQ3).

Then, you can use prepared scripts in the [runners/](runners/) directory to run experiments from the generated configs for a specific scenario or run them all. You can also run manually certain config by calling:
```
python main.py --config-path <configPath> --config-name <configName>
```

## Running experiments on real-world data
Directory [datasets/](datasets/) contains datasets of base classiffier outputs on real-world datasets.
Firstly, edit [main_real.py](main_real.py) (line 10) to select what dataset you want to use. Then, run the experiment by calling:
```
python main_real.py
```
