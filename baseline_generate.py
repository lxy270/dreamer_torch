
import argparse
from ruamel.yaml import YAML
yaml = YAML(typ='safe', pure=True)
import pathlib
import sys
import os
import pathlib
import torch
from dreamer import Dreamer, make_env
from parallel import Damy
import tools
import random
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)  # Python的随机种子
    np.random.seed(seed)  # Numpy的随机种子
    torch.manual_seed(seed)  # PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # PyTorch的GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True  # 让CuDNN确定性
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN加速优化（确保复现性）


# def data_processing(data_path, input_dim, action_dim, start,pred_horizen, device):
#     # data, _, _ = load_data_from_folder(data_path)
#     inputs, actions, targets, _, _ = load_data(data_path, flatten = False)
#     inputs = inputs[:, start:start+pred_horizen, :].to(device).float()
#     actions = actions[:,start:start+pred_horizen, :].to(device).float()
#     targets = targets[:, start:start+pred_horizen, :].to(device).float()
    
#     # 创建形状为 (batch_size, pred_horizen, 1) 的 is_first 张量
#     is_first = torch.zeros(inputs.shape[0], pred_horizen, 1, dtype=torch.bool).to(device)
#     is_first[:, 0, :] = True  # 设置每个序列的第一个时间步为 True，其余为 False

#     data_dict = {
#         "inputs": inputs,
#         "actions": actions,
#         "targets": targets,
#         "is_first": is_first
#     }
#     return data_dict

# python baseline_generate.py --configs dmc_proprio --task dmc_cheetah_run --logdir ./logdir/dmc_cheetah_run_random
# python baseline_generate.py --configs dmc_proprio --task dmc_reacher_hard --logdir ./logdir/dmc_reacher_hard_random
# python baseline_generate.py --configs dmc_proprio --task dmc_hopper_hop --logdir ./logdir/hopper_hop_random
# python baseline_generate.py --configs dmc_proprio --task dmc_panda_grasp --logdir ./logdir/dmc_panda_grasp
# python baseline_generate.py --configs dmc_proprio --task dmc_cheetah_run --logdir ./logdir/dmc_cheetah_run
# python baseline_generate.py --configs dmc_proprio --task dmc_humanoid_walk --logdir ./logdir/dmc_humanoid_walk
# python baseline_generate.py --configs dmc_proprio --task dmc_acrobot_swingup --logdir ./logdir/dmc_acrobot_swingup


def load_dreamers(config, dreamer_ckpt_path, seed = 187299):
    random.seed(seed)
    # 随机生成10个种子
    seeds = random.sample(range(1, 100000), 10)  # 从1到100000中随机抽取10个不重复的种子
    agents = []

    for seed in seeds:
        set_random_seed(seed)  # 使用当前种子
        logdir = pathlib.Path(config.logdir).expanduser()
        logdir.mkdir(parents=True, exist_ok=True)
        make = lambda mode, id: make_env(config, mode, id)
        env = [make("train", i) for i in range(config.envs)]
        env = [Damy(env) for env in env]

        # 获取动作空间维度并将其存储在 config 中
        acts = env[0].action_space
        print("Action Space", acts)
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

        agent = Dreamer(
            env[0].observation_space,
            env[0].action_space,
            config,
            None,
            None,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
       
        # 加载checkpoint（如果存在）
        if (logdir / dreamer_ckpt_path).exists():
            checkpoint = torch.load(logdir / dreamer_ckpt_path)
            agent.load_state_dict(checkpoint["agent_state_dict"],strict=False)
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
            agent._should_pretrain._once = False
        agents.append(agent)
        
    return agents


    


def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

# def load_configs(args):
#     configs = yaml.load(
#         (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text(),
#         Loader=yaml.FullLoader
#     )
#     name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
#     defaults = {}
#     for name in name_list:
#         recursive_update(defaults, configs[name])
#     return defaults


def load_configs(args):
    config_path = pathlib.Path(args.configs[0])  # 获取第一个传入的config路径
    configs = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)  # 读取YAML文件内容
    
    name_list = ["defaults"]
    defaults = {}
    for name in name_list:
        if name in configs:
            recursive_update(defaults, configs[name])
    
    return defaults


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    evaluate_dreamer(parser.parse_args(remaining))


# python baseline_generate.py --configs /home/chenjiehao/projects/dreamerv3-torch/configs.yaml --data_path /home/chenjiehao/projects/Neural-Simulator/data/cheetah/999_1.npz


#  11.657615661621094   16
# 10.820457458496094    10
# 9.042348861694336     5


# 2.415370464324951    16
#  1.8949111700057983  10
# 1.4691059589385986   5