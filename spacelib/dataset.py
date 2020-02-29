import os, random, string
import numpy as np
import torch
import numba
import gym

from collections import namedtuple

from .buffers import ArrayCollection, ArraySpec
from .flatter import Flatter

import pdb

def random_string(k):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=k))

def specs_from_space(fspace):
    specs = []
    for space in fspace:
        if isinstance(space, gym.spaces.Box):
            if np.prod(space.shape) > 1000 and np.mean(space.high) == 255: # "image classifier" 
                specs.append(ArraySpec(space.shape, np.uint8))
            else:
                specs.append(ArraySpec(space.shape, np.float32))
        elif isinstance(space, gym.spaces.Discrete):
            specs.append(ArraySpec(space.n, np.float32))
        elif isinstance(space, gym.spaces.MultiDiscrete):
            specs.append(ArraySpec(int(space.nvec.sum()), np.float32))
        else:
            raise ValueError
    return specs

@numba.jit(numba.float32(numba.float32[:], numba.float32))
def discount(rewards, gamma):
    c0 = 0.0
    for x in rewards[::-1]:
        c0 = x + gamma*c0
    return c0

@numba.jit(numba.float32[:](numba.float32[:], numba.float32, numba.int64))
def multidiscount(rewards, gamma, k):
    res = np.zeros(k, dtype=np.float32)
    c0 = 0.0
    ix = len(rewards)-1
    for x in rewards[::-1]:
        c0 = x + gamma * c0
        if ix < k:
            res[ix] = c0
        ix -= 1
    return res

Sequence = namedtuple(
    'Sequence',
    ['obs', 'act', 'rew', 'done', 'val', 'hidden'],
    defaults=(None, )*6
)

class Episode(ArrayCollection):
    def __init__(self, observation_space, action_space):
        self.obsf = Flatter(observation_space)
        self.actf = Flatter(action_space)
        n_obs = len(self.obsf)
        n_act = len(self.actf)
        self.obs_slice = 0 if self.obsf._singleton else slice(0, n_obs)
        self.act_slice = n_obs if self.actf._singleton else slice(n_obs, n_obs+n_act)
        self.rew_slice = n_obs+n_act
        self.done_slice = n_obs+n_act+1
        # import pdb; pdb.set_trace()
        
        super().__init__([*specs_from_space(self.obsf),
                         *specs_from_space(self.actf),
                         ArraySpec((), np.float32),
                         ArraySpec((), np.float32)])

    def append(self, step_data):
        ''' Add observation, action, reward, done for a single step. '''
        obs, act, rew, done = step_data
        # import pdb; pdb.set_trace()
        super().append([*self.obsf.flatten(obs), *self.actf.flatten(act), rew, done])
    
    def set_episode_data(self, episode_data):
        ''' Set the whole episodes's worth of (observation, action, reward, done). '''
        ''' Add observation, action, reward, done for a single step. '''
        obs, act, rew, done = episode_data
        for k, array_data in enumerate([*self.obsf.flatten(obs), *self.actf.flatten(act), rew, done]):
            self.setData(array_ix=k, data=array_data)

    def __getitem__(self, ix):
        res = super().__getitem__(ix)
        return res[self.obs_slice], res[self.act_slice], res[self.rew_slice], res[self.done_slice]
    
    def get_unflat(self, ix):
        obs, act, rew, done = self[ix]
        return self.obsf.unflatten(obs), self.actf.unflatten(act), rew, done

    def get_tensor(self, ix, batch_size=None, device=None):
        res = [torch.from_numpy(a).float().to(device) for a in super()[ix]]
        return res[self.obs_slice], res[self.act_slice], res[self.rew_slice], res[self.done_slice]



class RecurrentHidden(ArrayCollection):
    def __init__(self, hidden_dims):
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.hidden_dims = hidden_dims
        if not isinstance(hidden_dims, tuple):
            raise ValueError("hidden_dims should be int or tuple.")
        super().__init__([ArraySpec(hd, np.float32) for hd in hidden_dims])

    def get_tensor(self, ix, batch_size=None, device=None):
        return [torch.from_numpy(a).float().to(device) for a in self[ix]]

def collate_seq(batch, device=None):
    obs, act, rew, done, val, hidden = [],[],[],[],[],[]
    for seq in batch:
        obs.append(seq.obs)
        act.append(seq.act)
        rew.append(seq.rew)
        done.append(seq.done)
        if seq.val is not None:
            val.append(seq.val)
        if seq.hidden is not None:
            hidden.append(seq.hidden)

    return Sequence(
        obs = ([torch.from_numpy(np.stack(o)).float().to(device) for o in zip(*obs)]
                if not isinstance(obs[0], np.ndarray)
                    else torch.from_numpy(np.stack(obs)).float().to(device)),
        act = ([torch.from_numpy(np.stack(a)).float().to(device) for a in zip(*act)]
                if not isinstance(act[0], np.ndarray)
                    else torch.from_numpy(np.stack(act)).float().to(device)),
        rew = torch.from_numpy(np.stack(rew)).float().to(device),
        done = torch.from_numpy(np.stack(done)).float().to(device),
        val = torch.from_numpy(np.stack(val)).float().to(device) if len(val) > 0 else None,
        hidden = [torch.from_numpy(np.stack(h)).float().to(device) for h in zip(*hidden)] if len(hidden)>0 else None
    )

class RecurrentReplayBuffer:
    def __init__(self, observation_space, action_space, max_num_episodes=None, hidden_dim=None, data_root='data_tmp'):

        self.observation_space = observation_space
        self.action_space = action_space

        self.episode_rewards = []
        self.episodes = []
        self.hiddens = []
        
        self.hidden_dim = hidden_dim
        self.live_episode = None

        self.max_num_episodes = max_num_episodes

        self.data_root=data_root

    def __len__(self):
        return len(self.episodes)

    def add_episode(self, episode):
        self.episodes.append(episode)
        self.episode_rewards.append(episode[:][-2].sum())
        
        if self.hidden_dim is not None:
            new_hidden = RecurrentHidden(self.hidden_dim)
            new_hidden.allocate(len(episode))
            self.hiddens.append(new_hidden)
            del new_hidden

        if (self.max_num_episodes is not None) and (len(self) > self.max_num_episodes):
            worst_episode = np.argmin(self.episode_rewards)
            # worst_episode = 0
            del self.episodes[worst_episode]
            del self.episode_rewards[worst_episode]
            if self.hidden_dim is not None:
                del self.hiddens[worst_episode]

    def begin_episode(self, max_length=int(1e4)):
        if self.live_episode is not None:
            raise ValueError("End live episode before beginning a new one.")
        self.live_episode = Episode(self.observation_space, self.action_space)
        self.live_episode.allocate(max_length)

    def end_episode(self):
        save_dir = os.path.join(self.data_root, random_string(10))
        self.live_episode.toDisk(root=save_dir)
        self.add_episode(self.live_episode)
        self.live_episode = None

    def append(self, step_data):
        if self.live_episode is None:
            raise ValueError("Begin a new episode before adding experience.")
        self.live_episode.append(step_data)

    def sample_sequence(self, length, batch_size=None, gamma=None, hidden=True, device=None):
        tmp = [(k, len(e)) for k, e in enumerate(self.episodes) if len(e)>length]
        if len(tmp) == 0:
            return None

        to_sample = random.choices(tmp, weights=[t[1] for t in tmp], k=(batch_size or 1))

        batch = []
        for ep_no, ep_len in to_sample:
            start_ix = random.randrange(ep_len-length)
            sample_slice = slice(start_ix, start_ix+length)

            obs, act, rew, done = self.episodes[ep_no][sample_slice]

            if bool(hidden) and self.hidden_dim is not None:
                hidden_samples = self.hiddens[ep_no][sample_slice]
            else:
                hidden_samples = None

            if gamma is not None:
                rew_to_end = self.episodes[ep_no][start_ix:][-2]
                # import pdb; pdb.set_trace()
                val = multidiscount(rew_to_end, np.float32(gamma), length)
            else:
                val = None
            batch.append(Sequence(obs=obs, act=act, rew=rew, done=done, val=val, hidden=hidden_samples))
        if batch_size is None:
            return batch[0]
        else:
            return collate_seq(batch, device=device)

    def iter_sample(self, length, batch_size, minibatch_size=1, gamma=None, hidden=True, device=None):
        count = 0
        while count < batch_size:
            n = min(minibatch_size, batch_size-count)
            yield self.sample_sequence(length, batch_size=n, gamma=gamma, hidden=hidden, device=device)
            count += minibatch_size
