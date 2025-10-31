import argparse
import functools
import os
import pathlib
import sys
from datetime import datetime, date
sys.path.append('/home/chenjiehao/projects/dreamerv3_torch_ver/')
from torch.utils.tensorboard import SummaryWriter
# os.environ['MUJOCO_GL'] = 'egl'
# # 设置新的显示号
# os.environ['DISPLAY'] = ':4'
# os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ='safe', pure=True)

sys.path.append(str(pathlib.Path(__file__).parent))
import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import wandb
import torch
from torch import nn
from torch import distributions as torchd
# python dreamer.py --configs dmc_proprio --task dmc_go2_run --logdir ./logdir/dmc_go2_run
# python dreamer.py --configs dmc_proprio --task dmc_hopper_hop --logdir ./logdir/hopper_hop_random_5000
# python dreamer.py --configs dmc_proprio --task dmc_panda_grasp --logdir ./logdir/dmc_panda_grasp
# python dreamer.py --configs dmc_proprio --task dmc_cheetah_run --logdir ./logdir/dmc_cheetah_run  4-5
# python dreamer.py --configs dmc_proprio --task dmc_reacher_hard --logdir /home/chenjiehao/projects/dreamerv3_torch_ver/logdir/bouncingball_6000/  4-3
# python dreamer.py --configs dmc_proprio --task dmc_cheetah_run --logdir ./logdir/dmc_cheetah_run_mixed
# python dreamer.py --configs dmc_proprio --task dmc_humanoid_walk --logdir ./logdir/dmc_humanoid_walk   7
# python dreamer.py --configs dmc_proprio --task dmc_cartpole_swingup --logdir /home/chenjiehao/projects/dreamerv3_torch_ver/logdir/dmc_cartpole_swingup_new
to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, discrete_action=-1):
        self.eval_tvar = False
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        # self._step = logger.step // config.action_repeat
        self._step = 0
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._logger = SummaryWriter(log_dir=self._config.logdir)
        self.update_best_ckpt = False
        self.scheduler = None # torch.optim.lr_scheduler.CosineAnnealingLR(self._wm._model_opt._opt, T_max=300, eta_min=1e-8)

        self.best_loss = torch.inf
        if self.eval_tvar:
            eval_path = config.offline_traindir + f"seq-{config.task}-{config.act_mode}.npz"
            eval_data =  dict(np.load(eval_path, allow_pickle=True))
            for k in eval_data.keys():
                if k == 'metadata':
                    continue
                eval_data[k] = eval_data[k][:100]
        else:
            eval_path = config.offline_traindir + f"seq-{config.task}-{config.act_mode}-test.npz"
            eval_data = np.load(eval_path)
        if discrete_action != -1:
            action = eval_data['action'].squeeze()
            onehot = np.zeros((*action.shape, discrete_action), dtype=np.float32)
            idx = np.indices(action.shape) 
            onehot[(*idx, action)] = 1
            eval_action = onehot
        else:
            eval_action = eval_data['action'][:, :, None]
        is_first = np.zeros_like(eval_data['action'][:, :, None])
        is_first[:, 0] = 1
        self.eval_data = {'actions': torch.tensor(eval_action, device='cuda:0'),
                          'is_first': torch.tensor(is_first, device='cuda:0'),}
        if config.nq != 0:
            self.eval_data['targets'] = {'position': torch.tensor(eval_data['obs'][:, :, :config.nq], device='cuda:0'), 
                                         'velocity': torch.tensor(eval_data['obs'][:, :, config.nq:], device='cuda:0')}
        else:
            self.eval_data['targets'] = {'state': torch.tensor(eval_data['obs'], device='cuda:0', dtype=torch.float32),}
        self.eval_target = torch.tensor(eval_data['obs'], device='cuda:0')


    def __call__(self, obs, reset, state=None, training=True):
        if self.eval_tvar:
            condition_steps = 10
            state_prediction, _ = self._wm.propiro_pred(self.eval_data, condition_steps=condition_steps)
            # eval_loss = torch.nn.MSELoss()(state_prediction, self.eval_target[:, condition_steps:, :])
            eval_loss = torch.nn.functional.mse_loss(state_prediction, self.eval_target[:, condition_steps:, :], reduction="none")
            print('eval shape', eval_loss.shape)

            tvar = [1, 5, 10, 100, eval_loss.shape[1]]
            tloss = []
            for idx in tvar:
                tloss.append(eval_loss[:, :idx].mean())
            print('tloss ', tloss)
            savepath = f'/scorpio/home/yubei-stu-2/smallworld/results_tvar/dreamer-{self._config.task}-{self._config.act_mode}.pt'
            torch.save({'tvar': tvar, 'tloss': tloss}, savepath)
            exit()

        step = self._step
        if training:
            if self._update_count % 1000 == 0:
                condition_steps = 10
                state_prediction, _ = self._wm.propiro_pred(self.eval_data, condition_steps=condition_steps)
                eval_loss = torch.nn.MSELoss()(state_prediction, self.eval_target[:, condition_steps:, :])
                wandb.log({'eval_loss': eval_loss}, step=self._update_count)
                if self.best_loss > eval_loss:
                    self.best_loss = eval_loss
                    wandb.log({'best_img_loss': self.best_loss}, step=self._update_count)
                    self.update_best_ckpt = True

            steps = 200 # 100
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self.scheduler:
                self.scheduler.step()

            if True:
                for name, values in self._metrics.items():
                    # 记录每个 metric 的标量值到 TensorBoard
                    self._logger.add_scalar(name, float(np.mean(values)), self._update_count)
                    wandb.log({name: float(np.mean(values))}, step=self._update_count)
                    self._metrics[name] = []  # 重置 metrics

                # 如果启用了 video 预测日志
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.add_video("train_openl", to_np(openl), global_step=self._update_count)

                self._logger.flush()  # 确保日志及时写入

            

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        if len(data['action'].shape) < 3:
            data['action'] = data['action'][..., None]
        metrics = {}
        post, context, mets = self._wm._train(data, self._update_count)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    project_dir = f"{config.task}-{config.act_mode}-{config.comment}"
    logdir = pathlib.Path(config.logdir + project_dir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    wandb.init(
        project='dreamer_torch_sw',
        name=project_dir,  # 指定项目名
        config=config,
        settings=wandb.Settings(      # 不记录system status
            _disable_stats=True,      # 禁用系统状态监控
            _disable_meta=True        # 禁用元数据收集
        )
    )
    import gymnasium as gym
    discrete_action = -1
    if config.nq != 0:
        config.num_actions = 1
        action_space = gym.spaces.Box(-1, 1, dtype=np.float32)
        obs_space = gym.spaces.Dict({
                            "position": gym.spaces.Box(-np.inf, np.inf, (config.nq,), dtype=np.float32),
                            "velocity": gym.spaces.Box(-np.inf, np.inf, (config.nv,), dtype=np.float32)
                        })
    elif 'PandaPush' in config.task:
        import panda_gym
        config.num_actions = 3
        env = gym.make(f"PandaPush-v3")
        print('dt', env.unwrapped.sim.dt)
        print('substep', env.unwrapped.sim.n_substeps)
        action_space = env.action_space
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (24,), dtype=np.float32),})
        print("obs_space ", obs_space)
    elif 'PandaStack' in config.task:
        import panda_gym
        config.num_actions = 4
        env = gym.make(f"PandaStack-v3")
        print('dt', env.unwrapped.sim.dt)
        print('substep', env.unwrapped.sim.n_substeps)
        action_space = env.action_space
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (43,), dtype=np.float32),})
        print("obs_space ", obs_space)
    elif 'go' in config.task:
        discrete_action = 361 # todo
        config.num_actions = 361
        action_space = gym.spaces.Box(low=0, high=1, shape=(361,), dtype=np.float32) # gym.spaces.Discrete(361)
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (363,), dtype=np.float32),})
        print("obs_space ", obs_space)
    elif 'Maze' in config.task:
        discrete_action = 6
        config.num_actions = 6
        action_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32) # gym.spaces.Discrete(6)
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),})
        print("maze obs_space ", obs_space)
    elif 'Point3D' in config.task:
        discrete_action = 6
        config.num_actions = 6
        action_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32) # gym.spaces.Discrete(6)
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32),})
        print("point3d obs_space ", obs_space)
    else:
        import ale_py
        discrete_action = config.num_actions = 18
        env = gym.make('ALE/' + config.task, obs_type="ram")
        action_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32) # env.action_space
        obs_space = gym.spaces.Dict({'state': env.observation_space,})
        # Box(0, 255, (210, 160, 3), uint8) for atari image


    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes_single(config.offline_traindir + f"seq-{config.task}-{config.act_mode}.npz",
                                            nq=config.nq, limit=config.dataset_size, discrete_action=discrete_action)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=100000)
    # make = lambda mode, id: make_env(config, mode, id)
    # train_envs = [make("train", i) for i in range(config.envs)]
    # eval_envs = [make("eval", i) for i in range(config.envs)]
    # if config.parallel:
    #     train_envs = [Parallel(env, "process") for env in train_envs]
    #     eval_envs = [Parallel(env, "process") for env in eval_envs]
    # else:
    #     train_envs = [Damy(env) for env in train_envs]
    #     eval_envs = [Damy(env) for env in eval_envs]
    acts = action_space
    print("Action Space", acts)

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        obs_space,
        action_space,
        config,
        logger,
        train_dataset,
        discrete_action=discrete_action,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    if (logdir / "latest.pt").exists():
        print("load latest ckpt from: ", logdir / "latest.pt")
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    if config.from_ckpt:
        checkpoint = torch.load(config.from_ckpt)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        print('ckpt load from ', config.from_ckpt)

    # make sure eval will be executed once after config.steps
    while True:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            # eval_policy = functools.partial(agent, training=False)
            # tools.simulate(
            #     eval_policy,
            #     eval_envs,
            #     eval_eps,
            #     config.evaldir,
            #     logger,
            #     is_eval=True,
            #     episodes=config.eval_episode_num,
            # )
            # if config.video_pred_log:
            #     video_pred = agent._wm.video_pred(next(eval_dataset))
            #     logger.video("eval_openl", to_np(video_pred))
        # print("Start training.")
        # state = tools.simulate(
        #     agent,
        #     train_envs,
        #     train_eps,
        #     config.traindir,
        #     logger,
        #     limit=config.dataset_size,
        #     steps=config.eval_every,
        #     state=state,
        # )
        agent(None, False)
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        # if agent._update_count >= config.checkpt_every:
        #     print(f"save ckpt at {agent._update_count}")
        #     torch.save(items_to_save, logdir / f"ckpt{agent._update_count}.pt")
        #     config.checkpt_every *= 2
        if agent._update_count >= 10000 and agent.update_best_ckpt:
            print(f"save best ckpt at {agent._update_count}")
            torch.save(items_to_save, logdir / f"best.pt")
            agent.update_best_ckpt = False
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


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
    main(parser.parse_args(remaining))
