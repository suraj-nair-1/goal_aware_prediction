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

### Training
parser.add_argument('--trainsteps', type=int, default=300000, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=32, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=30, metavar='L', help='Chunk size')
parser.add_argument('--beta', type=float, default=1, metavar='g', help='Global KL weight (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='', help='Learning rate') 
parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
parser.add_argument('--checkpoint-interval', type=int, default=1000, metavar='I', help='Checkpoint interval (episodes)')
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


# Initialise training environment and experience replay memory
#####################################################
env = Env(args.env, args.seed, args.max_episode_length, args.action_repeat)
if args.experience_replay is not '' and os.path.exists(args.experience_replay):
  D = torch.load(args.experience_replay)
else:
  print("NO DATA PROVIDED")
  assert(False)
#####################################################

# Initialise model parameters randomly
#####################################################
transition_model = TransitionModel(args.hidden_size, env.action_size, args.activation_function).to(device=args.device)
decoder = Decoder(args.hidden_size,  env.action_size, args.activation_function).to(device=args.device)
encoder = Encoder(args.hidden_size, args.activation_function, ch=6).to(device=args.device)

param_list = list(transition_model.parameters()) + list(decoder.parameters()) + list(encoder.parameters())
optimiser = optim.Adam(param_list, lr=args.learning_rate, eps=1e-4)

## Load model if path is provided
if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  transition_model.load_state_dict(model_dicts['transition_model'])
  try:
    decoder.load_state_dict(model_dicts['decoder'])
  except:
    decoder.load_state_dict(model_dicts['residual_model'])
  encoder.load_state_dict(model_dicts['encoder'])
  optimiser.load_state_dict(model_dicts['optimiser'])
#####################################################

# Main Loop
#####################################################
for s in tqdm(range(args.trainsteps)):
  traj_size = args.chunk_size
  # TRAIN MODEL
  # Draw Samples
  observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)
  ### SHAPES: (T, B, C, H, W), (T, B, A), (T, B), (T, B, 1)
    
  # Relabel Goals
  reached_goals = observations[-1:, :, :, :, :].repeat(observations.size(0), 1, 1, 1, 1)

  # Set observations to (s_t, s_g) and goal_observations to (s_g, s_g)
  observations[:,:, 3:,:,:] = reached_goals[:,:, :3, :, :]
  goal_observations = observations.clone() 
  goal_observations[:,:,:3,:,:]  = goal_observations[:,:,3:,:,:]

  # Encode Observations
  encoding = encoder(observations[:traj_size])
  mu, log_std = encoding[:, :, :args.hidden_size], encoding[:, :, args.hidden_size:]
  std = torch.exp(log_std)
  samples = torch.empty(mu.shape).normal_(mean=0,std=1).cuda()
  encoding = mu + std * samples
  klloss = 0.5 * torch.mean(mu**2 + std**2 - torch.log(std**2) - 1)

  # Compute Residual Target
  target = observations[:,:, :3,:,:] - observations[:,:, 3:,:,:]

  # Forward Predictions
  # How many steps into the future to predict
  predlen = min(int(s / 50000), 10)
  all_losses = []
  # For each starting index
  for sp in range(args.chunk_size - predlen):
    next_step = []
    next_step_encoding = encoding[sp:sp+1]
    next_step.append(next_step_encoding)
    for p in range(predlen):
      this_act = actions[sp+p:sp+p+1]
      next_step_encoding = transition_model(next_step_encoding, this_act)
      next_step.append(next_step_encoding)
    next_step = torch.cat(next_step)
    pred = decoder(next_step)

    if sp == 0:
      target_preds_logging = pred

    # MSE Reconstruction Loss
    all_losses.append(((target[sp:sp+1+predlen] - pred[:1+predlen])**2).mean())


  r_loss = torch.stack(all_losses).mean()
  optimiser.zero_grad()
  (r_loss + args.beta * klloss).backward()
  optimiser.step()
  if s % 50 == 0:
    print(r_loss, klloss)

  metrics['observation_loss'].append(r_loss.cpu().detach().numpy())
  metrics['trainsteps'].append(s)
    
  ## Logging Models/Images
  if (s % args.checkpoint_interval == 0):
    newpath = os.path.join(results_dir, str(s))
    os.makedirs(newpath, exist_ok=True)
    metrics['teststeps'].append(s)
    video_frames = []

    for p in range(predlen + 1):
      video_frames.append(make_grid(torch.cat([
                      observations[p,:5,:3,:,:].cpu().detach(), 
                      observations[0,:5,3:,:,:].cpu().detach(), 
                      target[p,:5,:,:,:].cpu().detach(),
                      target_preds_logging[p,:5,:,:,:].cpu().detach(),
                      ],dim=3), nrow=1).numpy() / 255.0 )
    

    write_video(video_frames, 'train_step%s' % s, newpath)  # Lossy compression
    lineplot(metrics['trainsteps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)

    torch.save({'transition_model': transition_model.state_dict(), 
                'decoder': decoder.state_dict(), 
                'encoder': encoder.state_dict(), 
                'optimiser': optimiser.state_dict()}, 
               os.path.join(results_dir, 'models_%d.pth' % s))
