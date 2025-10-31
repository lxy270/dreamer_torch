
import sys
sys.path.insert(0, '/home/chenjiehao/projects/dreamerv3_torch_ver')
import argparse
import tools
import os
from ruamel.yaml import YAML
yaml = YAML(typ='safe', pure=True)
import torch

from parallel import Damy

# sys.path.append('/home/chenjiehao/projects/dreamerv3_torch_ver')
from dreamer import Dreamer, make_env

def load_dreamer(task, ckpt_path, device):
    print("load_dreamer")
    config_path = 'configs.yaml'
    with open(config_path, 'r') as file:
        config = yaml.load(file)
    defaults = {}
    for name in ['defaults', 'dmc_proprio']:
        recursive_update(defaults, config[name])
    
    config = argparse.Namespace(**defaults)
    config.device = device
    config.task = task
    env = Damy(make_env(config, "train", 0))
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        None,
        None,
    )
    agent.requires_grad_(requires_grad=False)
    checkpoint = torch.load(ckpt_path,weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"],strict=False)
    # tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    agent._should_pretrain._once = False
    return agent, env

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

# if __name__ == "__main__":
#     print('nihao')
#     agent = load_dreamer(task = 'dmc_hopper_hop', dreamer_ckpt_path='/home/weyl/chenjie_projects/dreamerv3_torch_ver/logdir/dmc_hopper_hop_c/latest.pt')