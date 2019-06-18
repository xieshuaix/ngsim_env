import h5py, os, pdb
import numpy as np
import utils
import tensorflow as tf
import hyperparams
from julia_env.julia_env import JuliaEnv
from my_gaussian_gru_policy import myGaussianGRUPolicy

NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': 1,
    'trajdata_i101-22agents-0750am-0805am.txt': 1
}


def load_validate_data(file_path='',
                       act_keys=None,
                       file_names=None,
                       debug_size=None,
                       min_length=50,
                       normalize_data=True,
                       shuffle=False,
                       act_low=-1,
                       act_high=1,
                       clip_std_multiple=np.inf):
    # loading varies based on dataset type
    if file_path == '':
        raise ValueError('Invalid file_path: {}'.format(file_path))
    if file_names is None:
        raise ValueError('act_keys must be non-empty list')
    if act_keys is None:
        act_keys = ['accel', 'turn_rate_global']
    x, feature_names = utils.load_x_feature_names(file_path, utils.get_file_id(file_names))

    # no need to flatten
    obs = x
    act_idxs = [i for (i, n) in enumerate(feature_names) if n in act_keys]
    act = x[:, :, act_idxs]

    if normalize_data:
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )


def normalize(x, clip_std_multiple=np.inf):
    mean = np.mean(np.mean(x, axis=0), axis=0, keepdims=True)
    x = x - mean
    x_flatten = np.reshape(x, [-1, 66])
    std = np.std(x_flatten, axis=0, keepdims=True) + 1e-8
    up = std * clip_std_multiple
    lb = - std * clip_std_multiple
    x = np.clip(x, lb, up)
    x = x / std
    return x, mean, std


def normalize_range(x, low, high):
    low = np.array(low)
    high = np.array(high)
    mean = (high + low) / 2.
    half_range = (high - low) / 2.
    x = (x - mean) / half_range
    x = np.clip(x, -1, 1)
    return x


def build_policy(args, env, latent_sampler=None):
    policy = myGaussianGRUPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_dim=args.recurrent_hidden_dim,
        output_nonlinearity=None,
        learn_std=True
    )
    return policy


def get_ground_truth(feature_database_path, traj_data_file_name):
    # for single agent, path='../../data/trajectories/ngsim.h5'
    # for multi-agent, path='~/gym/agen/multi_agent_22/ngsim_22agents.h5'
    # build components
    # env, act_low, act_high = utils.build_ngsim_env(args, exp_dir, vectorize=True)
    # TODO: determine these programmatically
    act_low = np.array([-4, -0.15])
    act_high = np.array([4, 0.15])
    data = load_validate_data(feature_database_path,
                              act_low=act_low,
                              act_high=act_high,
                              min_length=200 + 50,
                              clip_std_multiple=10.0,
                              file_names=traj_data_file_name)
    return data


# TODO: check the difference between this and build_ngsim_env in utils.py
def build_ngsim_env(args,
                    exp_dir='/tmp',
                    alpha=0.001,
                    vectorize=False,
                    render_params=None,
                    videoMaking=False):
    # TODO: pass params as argument
    print('Building environment...')
    basedir = os.path.expanduser('~/.julia/packages/NGSIM/OPF1x/data/')
    file_paths = [os.path.join(basedir, 'trajdata_i101_trajectories-0750am-0805am.txt')]
    if render_params is None:
        render_params = dict(
            viz_dir=os.path.join(exp_dir, 'imitate/viz'),
            zoom=5.
        )
    env_params = dict(
        trajectory_filepaths=file_paths,
        H=200,
        primesteps=50,
        action_repeat=1,
        terminate_on_collision=False,
        terminate_on_off_road=False,
        render_params=render_params,
        n_envs=1,
        n_veh=1,
        remove_ngsim_veh=False,
        reward=0
    )
    # order matters here because multiagent is a subset of vectorized
    # i.e., if you want to run with multiagent = true, then vectorize must 
    # also be true
    env_id = 'MultiagentNGSIMEnv'

    env = JuliaEnv(
        env_id=env_id,
        env_params=env_params,
        using='AutoEnvs'
    )
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = None
    return env, low, high
