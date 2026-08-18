"""
env_humanoid.py
Humanoid-Walk dataset for state-space (qpos/qvel) world model training.

Data format (from collect_humanoid.py):
  qpos    : (N, L+1, 28) float32
  qvel    : (N, L+1, 27) float32
  actions : (N, L, 21)   float32
  rewards : (N, L)        float32
  dones   : (N, L)        bool

Usage:
  ds = HumanoidStateDataset('humanoid_walk_data/humanoid_walk_random.npz')
  obs, acts, target = ds.sample_batch(batch_size=256, K=5)
  # obs:    (B, 55) float32,   current state [qpos; qvel]
  # acts:   (B, K, 21) float32, action sequence
  # target: (B, 55) float32,   state after K steps
"""

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PATH = 'data/cheetah_run_state_intermix.npz' # 'data/cheetah_run_state_random_10000.npz' # 'data/humanoid_walk_state_random_1000x1000.npz'

class HumanoidStateDataset(Dataset):
    """
    Humanoid state-space dataset with variable horizon K.
    Concatenates qpos (28d) + qvel (27d) → 55d state vector.
    """

    # def __init__(self, data_path=DATA_PATH, normalize=False, train_data_path=None):
    #     print('dataset normalize: ', normalize)
    #     data = np.load(data_path)
    #     print(f'load data from {data_path}')
    #     qpos = data['qpos']      # (N, L+1, 28)
    #     qvel = data['qvel']      # (N, L+1, 27)
    #     self.pos_dim = qpos.shape[-1]
    #     self.vel_dim = qvel.shape[-1]
    #     self.actions = data['actions']   # (N, L, 21)
    #     self.rewards = data['rewards'] if 'rewards' in data.keys() else np.zeros(data['actions'].shape[:2], dtype=np.float32)
    #     self.dones   = data['dones']   if 'dones' in data.keys() else np.zeros(data['actions'].shape[:2], dtype=np.bool_)

    #     # 拼接 qpos + qvel → states
    #     self.states = np.concatenate([qpos, qvel], axis=-1).astype(np.float32)  # (N, L+1, 55)

    #     self.N, self.L_plus1, self.state_dim = self.states.shape
    #     self.L = self.L_plus1 - 1
    #     self.action_dim = self.actions.shape[-1]

    #     # 可选：标准化
    #     self.normalize = normalize
    #     if normalize:
    #         # ── 决定 stats 的来源 ──────────────────────
    #         # 若当前是 test/eval 数据，强制用 train 数据来算 mean/std，避免
    #         # train 和 eval 使用各自的 norm 造成的分布偏移。
    #         is_eval = ('test' in data_path) or ('eval' in data_path)
    #         if is_eval:
    #             if train_data_path is None:
    #                 train_data_path = 'data/weyl_humanoid_random_ar1/train_95pct/dpwm.npz' if 'rand' in data_path else 'data/weyl_humanoid_tdmpc2_ar10/train_95pct/dpwm.npz' # humanoid spec + weyl spec
    #             print(f'[eval mode] computing norm stats from train_data_path={train_data_path}')
    #             train_data = np.load(train_data_path)
    #             train_states_full = np.concatenate(
    #                 [train_data['qpos'], train_data['qvel']], axis=-1
    #             ).astype(np.float32)
    #             train_actions_full = train_data['actions']
    #             assert train_states_full.shape[-1] == self.state_dim, \
    #                 f"train/eval state_dim mismatch: {train_states_full.shape[-1]} vs {self.state_dim}"
    #             assert train_actions_full.shape[-1] == self.action_dim, \
    #                 f"train/eval action_dim mismatch: {train_actions_full.shape[-1]} vs {self.action_dim}"
    #             stats_states  = train_states_full
    #             stats_actions = train_actions_full
    #         else:
    #             stats_states  = self.states
    #             stats_actions = self.actions

    #         # ── state norm ────────────────────────────
    #         flat = stats_states.reshape(-1, self.state_dim)
    #         self.state_mean = flat.mean(axis=0)
    #         self.state_std  = flat.std(axis=0) + 1e-8
    #         self.states = (self.states - self.state_mean) / self.state_std

    #         # ── action norm ───────────────────────────
    #         flat_a = stats_actions.reshape(-1, self.action_dim)
    #         self.action_mean = flat_a.mean(axis=0)
    #         self.action_std  = flat_a.std(axis=0) + 1e-8
    #         self.actions = (self.actions - self.action_mean) / self.action_std

    #     print(f"HumanoidStateDataset: {self.N} trajs x {self.L} steps")
    #     print(f"  state_dim={self.state_dim}, action_dim={self.action_dim}")
    #     if normalize:
    #         src = 'train_data_path' if (normalize and (('test' in data_path) or ('eval' in data_path))) else 'self'
    #         print(f"  normalized: state/action mean/std (stats from {src})")

    def __init__(self, data_path=DATA_PATH, normalize=True):
        print('dataset normalize: ', normalize)
        data = np.load(data_path)
        print(f'load data from {data_path}')
        qpos = data['qpos']      # (N, L+1, 28)
        qvel = data['qvel']      # (N, L+1, 27)
        self.pos_dim = qpos.shape[-1]
        self.vel_dim = qvel.shape[-1]
        self.actions = data['actions']   # (N, L, 21)
        self.rewards = data['rewards'] if 'rewards' in data.keys() else np.zeros(data['actions'].shape[:2], dtype=np.float32)
        self.dones   = data['dones']   if 'dones' in data.keys() else np.zeros(data['actions'].shape[:2], dtype=np.bool_)

        # 拼接 qpos + qvel → states
        self.states = np.concatenate([qpos, qvel], axis=-1).astype(np.float32)  # (N, L+1, 55)

        self.N, self.L_plus1, self.state_dim = self.states.shape
        self.L = self.L_plus1 - 1
        self.action_dim = self.actions.shape[-1]

        # 可选：标准化
        self.normalize = normalize
        # 沿 (N, L+1) 两个维度展平后算 mean/std
        flat = self.states.reshape(-1, self.state_dim)
        self.state_mean = flat.mean(axis=0)
        self.state_std = flat.std(axis=0) + 1e-8
        flat_a = self.actions.reshape(-1, self.action_dim)
        self.action_mean = flat_a.mean(axis=0)
        self.action_std = flat_a.std(axis=0) + 1e-8
        
        if normalize:
            self.states = (self.states - self.state_mean) / self.state_std
            # print(f'state mean {self.state_mean}, state std {self.state_std}')
            self.actions = (self.actions - self.action_mean) / self.action_std
            # print(f'action mean {self.action_mean}')
            # print(f'action std {self.action_std}')

        print(f"HumanoidStateDataset: {self.N} trajs x {self.L} steps")
        print(f"  state_dim={self.state_dim}, action_dim={self.action_dim}")
        if normalize:
            print(f"  normalized: state mean/std, action mean/std")

    def __len__(self):
        return self.N * self.L

    def sample_batch(self, batch_size, K):
        max_t = self.L - K
        assert max_t > 0, f"K={K} too large for trajectory length {self.L}"

        traj = np.random.randint(self.N, size=batch_size)
        t = np.random.randint(max_t, size=batch_size)

        obs = torch.from_numpy(self.states[traj, t])
        target = torch.from_numpy(self.states[traj, t + K])
        acts = torch.from_numpy(
            np.stack([self.actions[traj[i], t[i]:t[i]+K] for i in range(batch_size)])
        )

        return {'obs': obs, 'actions': acts, 'target': target}

    def sample_batch_with_reward(self, batch_size, K):
        """Same as sample_batch but also returns reward sum over K steps."""
        max_t = self.L - K
        traj = np.random.randint(self.N, size=batch_size)
        t = np.random.randint(max_t, size=batch_size)

        obs = torch.from_numpy(self.states[traj, t])
        target = torch.from_numpy(self.states[traj, t + K])
        acts = torch.from_numpy(
            np.stack([self.actions[traj[i], t[i]:t[i]+K] for i in range(batch_size)])
        )
        rews = torch.from_numpy(
            np.stack([self.rewards[traj[i], t[i]:t[i]+K].sum() for i in range(batch_size)])
        )                                                                # (B,)

        return obs, acts, target, rews

    def unnormalize_state(self, state_tensor):
        """将标准化后的 state 还原为原始值 (用于 render)"""
        if not self.normalize:
            return state_tensor
        mean = torch.from_numpy(self.state_mean).to(state_tensor.device)
        std = torch.from_numpy(self.state_std).to(state_tensor.device)
        return state_tensor * std + mean
    
    def normalize_state(self, state_tensor):
        """将标准化后的 state 还原为原始值 (用于 render)"""
        if not self.normalize:
            return state_tensor
        mean = torch.from_numpy(self.state_mean).to(state_tensor.device)
        std = torch.from_numpy(self.state_std).to(state_tensor.device)
        return (state_tensor - mean) / std

    def unnormalize_action(self, action_tensor):
        if not self.normalize:
            return action_tensor
        mean = torch.from_numpy(self.action_mean).to(action_tensor.device)
        std = torch.from_numpy(self.action_std).to(action_tensor.device)
        return action_tensor * std + mean

    def normalize_action(self, action_tensor):
        if not self.normalize:
            return action_tensor
        mean = torch.from_numpy(self.action_mean).to(action_tensor.device).to(action_tensor.dtype)
        std  = torch.from_numpy(self.action_std).to(action_tensor.device).to(action_tensor.dtype)
        return (action_tensor - mean) / std

    def get_qpos_qvel(self, state_tensor, unnormalize=True):
        """从 55d state 向量拆出 qpos (28d) 和 qvel (27d)，用于 render"""
        if unnormalize:
            state_tensor = self.unnormalize_state(state_tensor)
        qpos = state_tensor[..., :self.pos_dim]
        qvel = state_tensor[..., self.pos_dim:]
        return qpos, qvel