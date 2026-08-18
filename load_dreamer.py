import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import argparse
import tools
import os
from ruamel.yaml import YAML
yaml = YAML(typ='safe', pure=True)
import torch
import gymnasium as gym
import numpy as np

from parallel import Damy

# sys.path.append('/home/chenjiehao/projects/dreamerv3_torch_ver')
from dreamer import Dreamer, make_env

def load_dreamer(task, ckpt_path, device, norm_data_path=None):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["agent_state_dict"]
    decoder_uses_state = any(
        key.endswith("heads.decoder._mlp.mean_layer.state.weight")
        for key in checkpoint_state
    )

    with open('/scorpio/home/yubei-stu-2/dreamerv3_torch_ver/configs.yaml', 'r') as file:
        config = yaml.load(file)
    defaults = {}
    for name in ['defaults', 'dmc_proprio']:
        recursive_update(defaults, config[name])
    
    config = argparse.Namespace(**defaults)
    config.device = device
    config.task = task
    if norm_data_path is not None:
        norm_data = np.load(norm_data_path)
        if "qpos" in norm_data and "qvel" in norm_data:
            norm_states = np.concatenate(
                [norm_data["qpos"], norm_data["qvel"]], axis=-1
            ).astype(np.float32)
        elif "obs" in norm_data:
            norm_states = norm_data["obs"].astype(np.float32)
        elif "states" in norm_data:
            norm_states = norm_data["states"].astype(np.float32)
        else:
            raise KeyError(
                f"{norm_data_path} must contain qpos/qvel, obs, or states"
            )
        norm_flat = norm_states.reshape(-1, norm_states.shape[-1])
        config.obs_mean = norm_flat.mean(axis=0).tolist()
        config.obs_std = (norm_flat.std(axis=0) + 1e-8).tolist()
        print(
            f"Dreamer normalization: {norm_data_path} "
            f"({norm_states.shape[-1]} features)"
        )
    # env = Damy(make_env(config, "train", 0))
    if 'humanoid' in task:
        config.num_actions = 21
        acts = action_space = gym.spaces.Box(-np.inf, np.inf, (21,), dtype=np.float32)
        if decoder_uses_state:
            config.nq = config.nv = 0
            obs_space = gym.spaces.Dict({
                'state': gym.spaces.Box(-np.inf, np.inf, (55,), dtype=np.float32),
            })
        else:
            config.nq, config.nv = 28, 27
            obs_space = gym.spaces.Dict({
                'position': gym.spaces.Box(-np.inf, np.inf, (28,), dtype=np.float32),
                'velocity': gym.spaces.Box(-np.inf, np.inf, (27,), dtype=np.float32),
            })
    elif 'cheetah' in task:
        config.num_actions = 6
        acts = action_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        if decoder_uses_state:
            config.nq = config.nv = 0
            obs_space = gym.spaces.Dict({
                'state': gym.spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32),
            })
        else:
            config.nq = config.nv = 9
            obs_space = gym.spaces.Dict({
                'position': gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
                'velocity': gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            })

    agent = Dreamer(
        obs_space,
        action_space,
        config,
        None,
        None,
    )
    agent.requires_grad_(requires_grad=False)
    load_result = agent.load_state_dict(checkpoint["agent_state_dict"],strict=False)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)
    # tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    agent._should_pretrain._once = False
    return agent, None, load_result


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

# if __name__ == "__main__":
#     print('nihao')
#     agent = load_dreamer(task = 'dmc_hopper_hop', dreamer_ckpt_path='/home/weyl/chenjie_projects/dreamerv3_torch_ver/logdir/dmc_hopper_hop_c/latest.pt')
