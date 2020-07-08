# Goal-Aware Prediction: Learning to Model What Matters (GAP)

Code for the paper *Goal-Aware Prediction: Learning to Model what Matters. Suraj Nair, Silvio Savarese, Chelsea Finn. ICML 2020*

## Generate Data

First, generate a dataset using random exploration. 
```
python collect_data.py 
```

## Train Model

Train a model on the generated dataset
```
python train_model.py
```

## Evaluate Control with Model

Evaluate a trained model with MPC on different tasks. To use a pretrained model run
```
python eval_model.py 
```
