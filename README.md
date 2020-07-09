# Goal-Aware Prediction: Learning to Model What Matters (GAP)

Code for the paper [Goal-Aware Prediction: Learning to Model what Matters.](https://proceedings.icml.cc/static/paper_files/icml/2020/2981-Paper.pdf) Suraj Nair, Silvio Savarese, Chelsea Finn. *ICML 2020*

## Setup

Dependencies:
```
Python 3.6
torch
tqdm
plotly
python-opencv
metaworld
mujoco-py
```

Install the environments by running `pip install -e .` under the `gap_envs` directory. This depends on mujoco-py and Metaworld being installed.
Set the path to the gap_envs assets as 
`export ASSETS_PATH=<PATH>/gap_envs/assets/`

## Evaluate a Pretrained GAP

To run pretrained GAP models with planning on the control tasks, run `eval_gap.py` for example:
 
```
python eval_gap.py --id door_task_1 --task_name 1 --env GapBlock-v0 --hidden-size 256 --models pretrained_gap_models/gap_block.pth 
python eval_gap.py --id door_task_3 --task_name 3 --env GapDoor-v0 --hidden-size 256 --models pretrained_gap_models/gap_door.pth 
```

## Generate Data

To generate a new dataset using random exploration:
In the block env run
```
python collect_data.py --id data_block_env --env GapBlock-v0  --seed-episodes 2000
```
and in the door and blocks env run
```
python collect_data.py --id data_door_env --env GapDoor-v0  --seed-episodes 2000
```

## Train GAP

Train a GAP model on the generated dataset
```
python train_gap.py --id trained_gap_door --env GapDoor-v0 --experience-replay results/data_door_env/experience.pth --hidden-size 256 --batch-size 32 --chunk-size 30
```
