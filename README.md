# Goal-Aware Prediction: Learning to Model What Matters (GAP)

Code for the paper *Goal-Aware Prediction: Learning to Model what Matters. Suraj Nair, Silvio Savarese, Chelsea Finn. ICML 2020*

## Generate Data

First, generate a dataset using random exploration. 
For example in the block env run
```
python collect_data.py --id data_block_env --env OurEnvMany-v0  --seed-episodes 2000
```
and in the door and blocks env run
```
python collect_data.py --id data_door_env --env OurDoor-v0  --seed-episodes 2000
```

## Train Model

Train a GAP model on the generated dataset
```
python train_model.py --id gap_block_env --env OurEnvMany-v0 --experience-replay results/data_block_env/experience.pth --hidden-size 256 --batch-size 32 --chunk-size 30 --method gap
```

## Evaluate Control with Model

Evaluate a trained model with MPC on different tasks. To use a pretrained model run
```
python eval_model.py --id gap_block_control_task1 --task 1 --env OurEnvMany-v0 --hidden-size 256 --method gap --models=pretrained_model/gap_block_env/models_299000.pth 
```
