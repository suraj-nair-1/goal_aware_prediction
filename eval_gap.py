import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import Env
from memory import ExperienceReplay
from model import  Encoder, TransitionModel, Decoder
from utils import lineplot, write_video

import pickle

import cv2

#####################################################################
### General
parser = argparse.ArgumentParser(description='GAP')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--task_name', type=str, default="easy", metavar='H', help='Task ID')


### Env Specific
parser.add_argument('--env', type=str, default='Pendulum-v0', help='Gym environment')
parser.add_argument('--max-episode-length', type=int, default=100, metavar='T', help='Max episode length')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')

### Memory
parser.add_argument('--experience-size', type=int, default=100000, metavar='D', help='Experience replay size')
parser.add_argument('--seed-episodes', type=int, default=2000, metavar='S', help='Seed episodes')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')

### Architecture
parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')

### Load Trained Model
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

  
# Setup
results_dir = os.path.join('results', args.id)
os.makedirs(results_dir, exist_ok=True)
logdir =  os.path.join(results_dir, "logs")
os.makedirs(logdir, exist_ok=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')
metrics = {'trainsteps': [], 'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'transition_loss': [], 'kl_loss': [], 'teststeps':[]}

def run_trial(test_env, encoder, transition_model, decoder, trpath, args):
  observation = test_env.reset().cuda()
  total_reward = 0
  fr = []
  
  ## Replanning Iterations
  for ptw in range(2):
    ## Goal state
    goal = observation.clone()
    goal[:,:3,:,:]  = goal[:,3:,:,:] 

    ## Encode current state and goal state
    encoding = encoder(observation.unsqueeze(1)) 
    encoding = encoding[:, :, :args.hidden_size]
    goal_encoding = encoder(goal.unsqueeze(1))
    goal_encoding = goal_encoding[:, :, :args.hidden_size]
    
    ph = 15
    numsamples = 1000
    ## CEM Iterations
    for itr in range(3):
      itrpath = os.path.join(trpath, "iter_"+str(ptw)+"_"+str(itr))
      os.makedirs(itrpath, exist_ok=True)
      if itr == 0:
        ## Generate action samples 
        action_samples = []
        for n in range(numsamples):
          action_trajs = []
          r = np.random.randint(test_env.action_size)
          for j in range(ph):
            action_trajs.append(test_env.sample_random_action())
          action_trajs = torch.stack(action_trajs)
          action_samples.append(action_trajs)
        action_samples = torch.stack(action_samples).cuda()
      else:
        sortid = costs.argsort()
        actions_sorted = action_samples[sortid]
        actions_ranked = actions_sorted[:10]

        ## Refitting to Best Trajs
        mean, std = actions_ranked.mean(0), actions_ranked.std(0)
        smp = torch.empty(action_samples.shape).normal_(mean=0, std=1).cuda()
        mean = mean.unsqueeze(0).repeat(numsamples, 1, 1)
        std = std.unsqueeze(0).repeat(numsamples, 1, 1)
        action_samples = smp * std + mean

      curr_states = encoding.repeat(numsamples, 1, 1)
      all_states = []
      all_states.append(curr_states)
      
      ## Forward Predictions and Cost Evaluation
      for j in range(ph):
        next_states = transition_model(curr_states, action_samples[:, j].unsqueeze(1))
        curr_states = next_states
        all_states.append(curr_states)
      all_states = torch.stack(all_states)
#       all_res = decoder(all_states.squeeze())
      costs = ((goal_encoding.repeat(numsamples, ph+1, 1) - all_states.squeeze().permute(1,0,2))**2).mean(dim=(1,2))
      
    _, best_traj = costs.min(0)
    acts = action_samples[best_traj]

    ## Step in env
    for j in range(ph):
      next_observation, reward, done = test_env.step(acts[j].cpu())
      reward /= args.action_repeat
      total_reward += reward
      fr.append(next_observation[:,:3].squeeze().cpu().detach().numpy() / 255.0 )
    observation = next_observation.cuda()
  write_video(fr, str(total_reward)+'rollout', trpath)
  ## Check Success
  success = test_env._env.is_goal()
  return success



env = Env(args.env, args.seed, args.max_episode_length, args.action_repeat)
# Initialise model parameters randomly
#####################################################
transition_model = TransitionModel(args.hidden_size, env.action_size, args.activation_function).to(device=args.device)
decoder = Decoder(args.hidden_size,  env.action_size, args.activation_function).to(device=args.device)
encoder = Encoder(args.hidden_size, args.activation_function, ch=6).to(device=args.device)


## Load model if path is provided
if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  transition_model.load_state_dict(model_dicts['transition_model'])
  try:
    decoder.load_state_dict(model_dicts['decoder'])
  except:
    decoder.load_state_dict(model_dicts['residual_model'])
  encoder.load_state_dict(model_dicts['encoder'])
#####################################################


transition_model.eval()
decoder.eval()
encoder.eval()

## If testing run planning
test_env = Env(args.env, args.seed, args.max_episode_length, args.action_repeat, problem=args.task_name)
num_trials = 200
with torch.no_grad():
  newpath = os.path.join(results_dir, "planning")
  os.makedirs(newpath, exist_ok=True)
  rewards = []
  for tr in range(num_trials):
    trpath = os.path.join(newpath, "planning_"+str(tr))
    os.makedirs(trpath, exist_ok=True)

    ## If running planning trials
    success = run_trial(test_env, encoder, transition_model, decoder, trpath, args)
    rewards.append(success)
    print("******", np.mean(rewards))
  metrics["test_rewards"].append(np.mean(rewards))
  test_env.close()
lineplot(metrics['teststeps'][-len(metrics['test_rewards']):], metrics['test_rewards'], 'test_rewards', results_dir)
