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
        # Previous per-feature statistics from cheetah_run_state_random.npz:
        self.obs_mean = [
            -0.8466714024543762, -0.11200540512800217,
            0.05405983328819275, 0.0002913616772275418,
            0.038189925253391266, -0.03948764503002167,
            -0.081931933760643, -0.10421689599752426,
            -0.09087486565113068, -0.17118185758590698,
            -2.5322346118628047e-05, 0.001493218820542097,
            0.0010638055391609669, 0.004946540575474501,
            -0.0028293728828430176, -0.013455092906951904,
            -0.012141266837716103, -0.01069149561226368,
        ]
        self.obs_std = [
            0.6688496470451355, 0.034241463989019394,
            0.06354093551635742, 0.15606531500816345,
            0.1735278069972992, 0.18075938522815704,
            0.09739918261766434, 0.16425222158432007,
            0.19257119297981262, 0.5030624270439148,
            0.45922034978866577, 1.1172008514404297,
            3.821852207183838, 5.003693580627441,
            5.02598762512207, 2.693417549133301,
            3.884706974029541, 4.176016330718994,
        ]

        # Active per-feature statistics from
        # /scorpio/home/yubei-stu-2/tcond/data/cheetah_run_state_intermix.npz.
        # These follow env_humanoid.py: concatenate [qpos, qvel], flatten the
        # trajectory/time axes, then compute per-feature mean and std (+ 1e-8).
        # self.obs_mean = [
        #     20.37139129638672, -0.07417750358581543,
        #     0.5046687722206116, -0.06243321672081947,
        #     -0.04251820966601372, -0.07586397230625153,
        #     -0.22625090181827545, 0.0069607398472726345,
        #     -0.06271885335445404, 4.119309902191162,
        #     0.002694385591894388, 0.0753093734383583,
        #     0.007443673443049192, -0.008049629628658295,
        #     0.013028179295361042, -0.10203664749860764,
        #     0.03597825765609741, -0.006187621969729662,
        # ]
        # self.obs_std = [
        #     18.07161521911621, 0.12715467810630798,
        #     1.2095049619674683, 0.30905407667160034,
        #     0.38757970929145813, 0.30774742364883423,
        #     0.27587181329727173, 0.2588687837123871,
        #     0.23161032795906067, 3.484938144683838,
        #     0.7637878060340881, 1.7436968088150024,
        #     6.357306957244873, 8.448796272277832,
        #     7.873904705047607, 5.783249855041504,
        #     5.426361083984375, 4.461714744567871,
        # ]
        obs_dim = sum(
            int(np.prod(space.shape)) for space in obs_space.spaces.values()
        )
        if len(self.obs_mean) != obs_dim or len(self.obs_std) != obs_dim:
            raise ValueError(
                f"Observation normalization has {len(self.obs_mean)} features, "
                f"but task {config.task!r} has {obs_dim}. Recompute obs_mean "
                "and obs_std for this training dataset."
            )
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
        self.scheduler = None # torch.optim.lr_scheduler.CosineAnnealingLR(self._wm._model_opt._opt, T_max=300, eta_min=1e-10)

        self.best_loss = torch.inf
        if self.eval_tvar:
            eval_path = config.offline_traindir + f"seq-{config.task}-{config.act_mode}.npz"
            eval_data =  dict(np.load(eval_path, allow_pickle=True))
            for k in eval_data.keys():
                if k == 'metadata':
                    continue
                eval_data[k] = eval_data[k][:100]
        else:
            if 'tcond' in config.comment:
                eval_path = config.offline_traindir + f"tcond-{config.task}-{config.act_mode}-test.npz"
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
        if 'humanoid' in config.task or 'reacher' in config.task:
            self.eval_data['actions'] = self.eval_data['actions'].to(torch.float32)
        if config.nq != 0:
            self.eval_data['targets'] = {'position': torch.tensor(eval_data['obs'][:, :, :config.nq], device='cuda:0'), 
                                         'velocity': torch.tensor(eval_data['obs'][:, :, config.nq:], device='cuda:0')}
        else:
            self.eval_data['targets'] = {'state': torch.tensor(eval_data['obs'], device='cuda:0', dtype=torch.float32),}
        self.eval_target = torch.tensor(eval_data['obs'], device='cuda:0')

    def _normalize_obs(self, data):
        """Return a shallow copy with only observation fields normalized."""
        normalized = dict(data)
        if self._config.nq != 0:
            keys_and_slices = (
                ("position", slice(None, self._config.nq)),
                ("velocity", slice(self._config.nq, None)),
            )
        else:
            keys_and_slices = (("state", slice(None)),)

        for key, feature_slice in keys_and_slices:
            if key not in normalized:
                continue
            value = normalized[key]
            if isinstance(value, torch.Tensor):
                mean = value.new_tensor(self.obs_mean[feature_slice])
                std = value.new_tensor(self.obs_std[feature_slice]).clamp_min(1e-6)
            else:
                mean = np.asarray(self.obs_mean[feature_slice], dtype=np.float32)
                std = np.maximum(
                    np.asarray(self.obs_std[feature_slice], dtype=np.float32),
                    1e-6,
                )
            normalized[key] = (value - mean) / std
        return normalized

    def _denormalize_prediction(self, prediction):
        mean = prediction.new_tensor(self.obs_mean)
        std = prediction.new_tensor(self.obs_std).clamp_min(1e-6)
        return prediction * std + mean


    def __call__(self, obs, reset, state=None, training=True):
        if self.eval_tvar:
            condition_steps = 10
            eval_data = dict(self.eval_data)
            eval_data["targets"] = self._normalize_obs(eval_data["targets"])
            state_prediction, _ = self._wm.propiro_pred(eval_data, condition_steps=condition_steps)
            state_prediction = self._denormalize_prediction(state_prediction)
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
                eval_data = dict(self.eval_data)
                eval_data["targets"] = self._normalize_obs(eval_data["targets"])
                state_prediction, _ = self._wm.propiro_pred(eval_data, condition_steps=condition_steps)
                state_prediction = self._denormalize_prediction(state_prediction)
                eval_loss = torch.nn.MSELoss()(state_prediction, self.eval_target[:, condition_steps:, :])
                eval_loss1 = torch.nn.MSELoss()(state_prediction[:, :1, :], self.eval_target[:, condition_steps:condition_steps+1, :])
                eval_loss16 = torch.nn.MSELoss()(state_prediction[:, :16, :], self.eval_target[:, condition_steps:condition_steps+16, :])
                wandb.log({'eval_loss1': eval_loss1}, step=self._update_count)
                wandb.log({'eval_loss16': eval_loss16}, step=self._update_count)
                wandb.log({'eval_loss90': eval_loss}, step=self._update_count)
                if self.best_loss > eval_loss:
                    self.best_loss = eval_loss
                    wandb.log({'best_img_loss90': self.best_loss}, step=self._update_count)
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
                    video_data = self._normalize_obs(next(self._dataset))
                    openl = self._wm.video_pred(video_data)
                    self._logger.add_video("train_openl", to_np(openl), global_step=self._update_count)

                self._logger.flush()  # 确保日志及时写入

            

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._normalize_obs(obs)
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
        data = self._normalize_obs(data)
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
    if config.nq != 0: # dm_phy
        config.num_actions = 1
        action_space = gym.spaces.Box(-1, 1, dtype=np.float32)
        obs_space = gym.spaces.Dict({
                            "position": gym.spaces.Box(-np.inf, np.inf, (config.nq,), dtype=np.float32),
                            "velocity": gym.spaces.Box(-np.inf, np.inf, (config.nv,), dtype=np.float32)
                        })
    elif 'PandaPush' in config.task: # changed for v2
        import panda_gym
        config.num_actions = 3
        env = gym.make(f"PandaPush-v3")
        print('dt', env.unwrapped.sim.dt)
        print('substep', env.unwrapped.sim.n_substeps)
        action_space = env.action_space
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32),})
        print("obs_space ", obs_space)
    elif 'PandaStack' in config.task:  # changed for v2
        import panda_gym
        config.num_actions = 4
        env = gym.make(f"PandaStack-v3")
        print('dt', env.unwrapped.sim.dt)
        print('substep', env.unwrapped.sim.n_substeps)
        action_space = env.action_space
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (31,), dtype=np.float32),})
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
    elif 'humanoid' in config.task:
        config.num_actions = 21
        action_space = gym.spaces.Box(-np.inf, np.inf, (21,), dtype=np.float32)
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (55,), dtype=np.float32),}) # 67 before tcond
    elif 'cheetah' in config.task:
        config.num_actions = 6
        action_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32),}) # for tcond
    elif 'reacher' in config.task:
        config.num_actions = 2
        action_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),})
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

    if 'tcond' in config.comment:
        train_path = config.offline_traindir + f"tcond-{config.task}-{config.act_mode}.npz"
    else:
        train_path = config.offline_traindir + f"seq-{config.task}-{config.act_mode}.npz"
    print(f'train data path: {train_path}')
    train_eps = tools.load_episodes_single(train_path,
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
