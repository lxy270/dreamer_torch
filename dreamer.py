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
        # self.obs_mean = [
        #     -0.8466714024543762, -0.11200540512800217,
        #     0.05405983328819275, 0.0002913616772275418,
        #     0.038189925253391266, -0.03948764503002167,
        #     -0.081931933760643, -0.10421689599752426,
        #     -0.09087486565113068, -0.17118185758590698,
        #     -2.5322346118628047e-05, 0.001493218820542097,
        #     0.0010638055391609669, 0.004946540575474501,
        #     -0.0028293728828430176, -0.013455092906951904,
        #     -0.012141266837716103, -0.01069149561226368,
        # ]
        # self.obs_std = [
        #     0.6688496470451355, 0.034241463989019394,
        #     0.06354093551635742, 0.15606531500816345,
        #     0.1735278069972992, 0.18075938522815704,
        #     0.09739918261766434, 0.16425222158432007,
        #     0.19257119297981262, 0.5030624270439148,
        #     0.45922034978866577, 1.1172008514404297,
        #     3.821852207183838, 5.003693580627441,
        #     5.02598762512207, 2.693417549133301,
        #     3.884706974029541, 4.176016330718994,
        # ]

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

        # Previous per-feature statistics from
        # /scorpio/home/yubei-stu-2/tcond/data/humanoid_walk_state_random.npz.
        # These follow env_humanoid.py: concatenate [qpos, qvel], flatten the
        # trajectory/time axes, then compute per-feature mean and std (+ 1e-8).
        # self.obs_mean = [
        #     0.03798120841383934, -0.10191689431667328, 0.1656177043914795,
        #     0.16686458885669708, 0.11508369445800781, 0.041589345782995224,
        #     0.09095874428749084, -0.012129470705986023, -0.13569609820842743,
        #     -0.007351782638579607, -0.12327157706022263, -0.12007284164428711,
        #     -0.35809946060180664, -1.4829778671264648, -0.07686090469360352,
        #     0.036160510033369064, -0.12280319631099701, -0.09503483027219772,
        #     -0.3688044846057892, -1.4749977588653564, -0.07525240629911423,
        #     -0.03074156492948532, -0.1411679983139038, -0.22698529064655304,
        #     -0.2202613651752472, 0.1628495752811432, 0.23772354423999786,
        #     -0.22593176364898682, -0.00019009882817044854,
        #     -0.00311070098541677, -0.04132433980703354,
        #     -0.027197321876883507, -0.08198679983615875,
        #     0.006752088665962219, -0.0014453640906140208, 0.1526574045419693,
        #     0.00211126240901649, -0.04470006003975868, -0.008729278109967709,
        #     -0.08281100541353226, 0.14131894707679749, 0.21066170930862427,
        #     0.05319307744503021, -0.05032213404774666,
        #     -0.008477555587887764, -0.08240807801485062,
        #     0.14124932885169983, 0.20589813590049744,
        #     -0.04544480890035629, -0.026485126465559006,
        #     0.06066042184829712, -0.1562371701002121, 0.030268296599388123,
        #     -0.06451356410980225, -0.1561702936887741,
        # ]
        # self.obs_std = [
        #     0.5701468586921692, 0.5747877955436707, 0.15801140666007996,
        #     0.4856375753879547, 0.4721255600452423, 0.48647502064704895,
        #     0.49382317066192627, 0.3982589840888977, 0.4173111021518707,
        #     0.36409929394721985, 0.1968085616827011, 0.48662489652633667,
        #     0.5936410427093506, 1.0145838260650635, 0.6960034370422363,
        #     0.7319065928459167, 0.19651220738887787, 0.48245373368263245,
        #     0.6066346764564514, 1.0170915126800537, 0.6966444253921509,
        #     0.7335487008094788, 0.8909797072410583, 0.7281660437583923,
        #     0.969681441783905, 0.8912369012832642, 0.7311902642250061,
        #     0.9708925485610962, 0.38478532433509827, 0.3855040669441223,
        #     0.5786380767822266, 1.7564221620559692, 2.198415517807007,
        #     3.873048782348633, 3.9398155212402344, 4.168158054351807,
        #     3.269749164581299, 3.1906933784484863, 3.85471248626709,
        #     6.962044715881348, 12.141858100891113, 18.019445419311523,
        #     19.137069702148438, 3.190166711807251, 3.8477375507354736,
        #     6.992808818817139, 12.250439643859863, 18.08917999267578,
        #     19.234926223754883, 8.003087043762207, 8.041154861450195,
        #     15.38257122039795, 8.113006591796875, 8.135037422180176,
        #     15.492866516113281,
        # ]

        # Active per-feature statistics from
        # /scorpio/home/yubei-stu-2/tcond/data/humanoid_walk_state_intermix.npz.
        self.obs_mean = [
            -1.4009170532226562, -1.0451563596725464, 0.5868939757347107,
            0.10669253021478653, 0.01900624856352806, -0.010634387843310833,
            0.08876361697912216, -0.24043427407741547, -0.1285434067249298,
            -0.08084239810705185, -0.1720033586025238, -0.5235905051231384,
            -0.5493985414505005, -0.9877043962478638, 0.07504254579544067,
            -0.1497340351343155, -0.13143125176429749, -0.09196905791759491,
            -0.7952171564102173, -1.554720163345337, -0.11703119426965714,
            0.05086883157491684, 0.14571064710617065, -0.2450660914182663,
            -0.6452822685241699, -0.11748659610748291, 0.6197184920310974,
            -0.16721822321414948, -0.16325508058071136,
            -0.07327321916818619, -0.042401980608701706,
            -0.1719816029071808, -0.03393366560339928,
            -0.02825571782886982, -0.0077944169752299786,
            0.06292422860860825, 0.00149207201320678, 0.0642504021525383,
            0.027089299634099007, -0.058333199471235275,
            0.07368064671754837, 0.06567860394716263, 0.14396709203720093,
            -0.08632451295852661, -0.005554706323891878,
            0.1005450189113617, 0.377289354801178, 0.2242233157157898,
            -0.11415280401706696, 0.01836523413658142, 0.03902703523635864,
            -0.06241389364004135, 0.07965965569019318,
            -0.010318524204194546, -0.1272175908088684,
        ]
        self.obs_std = [
            5.524777412414551, 4.6306891441345215, 0.4762152433395386,
            0.5775547027587891, 0.35671600699424744, 0.3645387291908264,
            0.6065661907196045, 0.4208100140094757, 0.3967120945453644,
            0.34185221791267395, 0.2154780477285385, 0.49150991439819336,
            0.7761805057525635, 1.057207465171814, 0.6903636455535889,
            0.7237813472747803, 0.21157212555408478, 0.5406605005264282,
            0.7621198892593384, 1.0960242748260498, 0.6760191917419434,
            0.7263838648796082, 0.8672032952308655, 0.7597022652626038,
            0.9852666854858398, 0.8587227463722229, 0.793128252029419,
            1.007764220237732, 0.9507737159729004, 0.8466731905937195,
            1.056557059288025, 2.054273843765259, 2.5906240940093994,
            3.8144919872283936, 3.60932993888855, 3.972778081893921,
            3.7023210525512695, 3.2424299716949463, 3.482320785522461,
            6.730387210845947, 11.24797534942627, 15.567275047302246,
            16.842321395874023, 3.0159897804260254, 3.5922157764434814,
            7.591004371643066, 12.764810562133789, 16.222307205200195,
            17.055946350097656, 7.830094814300537, 7.427765369415283,
            13.890138626098633, 8.392330169677734, 7.30751895904541,
            13.69578742980957,
        ]
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
