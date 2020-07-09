from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F



## f_dec, model of dynamics in latent space
class TransitionModel(nn.Module):
  __constants__ = ['min_std_dev']

  def __init__(self, hidden_size, action_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(hidden_size + action_size, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 128)
    self.fc4 = nn.Linear(128, hidden_size)
    
  def forward(self, prev_hidden, action):
    hidden = torch.cat([prev_hidden, action], dim=-1)
    trajlen, batchsize = hidden.size(0), hidden.size(1)
    hidden.view(-1, hidden.size(2))
    hidden = self.act_fn(self.fc1(hidden))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    hidden = self.fc4(hidden)
    hidden = hidden.view(trajlen, batchsize, -1)
    return hidden
  

## Image Encoder
class Encoder(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, hidden_size, activation_function='relu', ch=6):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.softmax = nn.Softmax(dim=2)
    self.sigmoid = nn.Sigmoid()
    self.ch = ch
    self.conv1 = nn.Conv2d(self.ch, 32, 4, stride=2) #3
    self.conv1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.conv4_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 2* hidden_size)

  def forward(self, observation):
    trajlen, batchsize = observation.size(0), observation.size(1)
    self.width = observation.size(3)
    observation = observation.view(trajlen*batchsize, 6, self.width, 64)
    if self.ch == 3:
      observation = observation[:, :3, :, :]

    
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv1_1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv2_1(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv3_1(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = self.act_fn(self.conv4_1(hidden))
    
    hidden = hidden.view(trajlen*batchsize, -1)
    hidden = self.act_fn(self.fc1(hidden))
    hidden = self.fc2(hidden)
    hidden = hidden.view(trajlen, batchsize, -1)
    return hidden
  

  
## Residual Model   
class Decoder(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, hidden_size, action_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(hidden_size * 1, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 128)
    
    self.conv1 = nn.ConvTranspose2d(128, 128, 5, stride=2)
    self.conv1_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv2_1 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv3_1 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
    self.conv4_1 = nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)

  def forward(self, hidden):
    trajlen, batchsize = hidden.size(0), hidden.size(1)
    hidden = hidden.view(trajlen*batchsize, -1)
    hidden = self.act_fn(self.fc1(hidden))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    hidden = hidden.view(-1, 128, 1, 1)

    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    residual = self.conv4(hidden)
    
    residual = residual.view(trajlen, batchsize, residual.size(1), residual.size(2), residual.size(3))
    return residual