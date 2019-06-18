import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os
import sys
import tensorflow as tf
import time

# visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import hgail.misc.simulation
import hgail.misc.utils
import hyperparams as hp
import utils
import rls, pdb
import validate_utils

plt.style.use("ggplot")


def online_adaption(env=None,
                    policy=None,
                    max_steps=1000,
                    obs=None,
                    mean=None,
                    render=False,
                    env_kwargs=None,
                    lbd=0.99,
                    adapt_steps=1,
                    output_dir=''):
    if env_kwargs is None:
        raise ValueError('env_kwargs unspecified')
    if len(obs.shape) == 2:
        obs = np.expand_dims(obs, axis=0)
        mean = np.expand_dims(mean, axis=0)

    theta = np.load('theta.npy')  # shape of (65, 2)
    theta = np.mean(theta)

    x = env.reset(**env_kwargs) # shape of (22, 66)

    n_agents = x.shape[0]
    predicted_trajs, adapnets = [], []
    policy.reset([True] * n_agents)
    prev_actions, prev_hiddens = None, None

    max_steps = min(max_steps, obs.shape[1])  # shape of (22, 331, 66), where 22 is number of agents

    mean = np.expand_dims(mean, axis=2)
    prev_hiddens = np.zeros([n_agents, 64])

    param_length = 65 if adapt_steps == 1 else 195

    for i in range(n_agents):
        adapnets.append(rls.rls(lbd, theta, param_length, 2))

    time_total = 0
    num_job = max_steps - 1
    fig_path = os.path.join(output_dir, 'adaption_train.png')
    plt.xlabel('Step')
    plt.ylabel('')
    for step in range(max_steps - 1):
        start = time.time()
        a, a_info, hidden_vec = policy.get_actions_with_prev(obs[:, step, :], mean[:, step, :], prev_hiddens)

        if adapt_steps == 1:
            adap_vec = hidden_vec
        else:
            adap_vec = np.concatenate((hidden_vec, prev_hiddens, obs[:, step, :]), axis=1)

        adap_vec = np.expand_dims(adap_vec, axis=1)

        for i in range(n_agents):
            adapnets[i].update(adap_vec[i], mean[i, step + 1, :])
            adapnets[i].draw.append(adapnets[i].theta[6, 1])

        prev_actions, prev_hiddens = a, hidden_vec

        traj = prediction(env_kwargs, obs[:, step + 1, :], adapnets, env, policy, prev_hiddens, n_agents, adapt_steps)

        predicted_trajs.append(traj)
        d = np.stack([adapnets[i].draw for i in range(n_agents)])
        end = time.time()
        time_total += end - start

        job_done = step + 1
        if job_done % max(int(num_job / 100), 5) == 0 or job_done == num_job:
            print('Step {}/{} ({:.0%}), ETA: {:.0f}s...'.format(step, num_job, job_done / num_job,
                                                                time_total / job_done * (num_job - job_done)),
                  end='\r')
            if output_dir != '':
                for i in range(n_agents):
                    plt.plot(range(step + 1), d[i, :])
                plt.savefig(fig_path)

    return predicted_trajs


def prediction(env_kwargs, x, adapnets, env, policy, prev_hiddens, n_agents, adapt_steps):
    traj = hgail.misc.simulation.Trajectory()
    predict_span = 200
    for i in range(predict_span):
        a, a_info, hidden_vec = policy.get_actions(x)

        if adapt_steps == 1:
            adap_vec = hidden_vec
        else:
            adap_vec = np.concatenate((hidden_vec, prev_hiddens, x), axis=1)

        means = np.zeros([n_agents, 2])
        log_std = np.zeros([n_agents, 2])
        for i in range(x.shape[0]):
            means[i] = adapnets[i].predict(np.expand_dims(adap_vec[i], 0))
            log_std[i] = np.log(np.std(adapnets[i].theta, axis=0))

        prev_hiddens = hidden_vec

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_std) + means

        nx, r, dones, e_info = env.step(actions)
        traj.add(x, actions, r, a_info, e_info)
        if any(dones):
            break
        x = nx

    # this should be delete and replaced
    y = env.reset(**env_kwargs)

    return traj.flatten()


def collect_trajectories(hyperparams=None, state=None, env_fn=None, policy_fn=None, egoids=None, starts=None,
                         traj_list=None, pid=None, max_steps=1000, use_hgail=False, random_seed=None,
                         lbd=0.99, adapt_steps=1, output_dir='', ground_truth=None):
    env, act_low, act_high = env_fn(hyperparams, alpha=0.0)
    policy = policy_fn(hyperparams, env)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # then load parameters
        if use_hgail:
            for i, level in enumerate(policy):
                level.algo.policy.set_param_values(state[i]['policy'])
            policy = policy[0].algo.policy
        else:
            policy.set_param_values(state['policy'])

        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = state['normalzing']['obs_mean']
            normalized_env._obs_var = state['normalzing']['obs_var']

        # collect trajectories
        n_ego = len(egoids)

        if not hyperparams.env_multiagent:
            sample = np.random.choice(ground_truth['observations'].shape[0], 2)

        kwargs = dict()
        if hyperparams.env_multiagent:
            # I add not because single simulation has no orig_x etc.
            if random_seed:
                kwargs = dict(random_seed=random_seed + egoid)
            traj = online_adaption(env=env,
                                   policy=policy,
                                   max_steps=max_steps,
                                   obs=ground_truth['observations'],
                                   mean=ground_truth['actions'],
                                   env_kwargs=kwargs,
                                   lbd=lbd,
                                   adapt_steps=adapt_steps,
                                   output_dir=output_dir)
            traj_list.append(traj)
        else:
            for i in sample:
                print('pid: {} traj: {} / {}'.format(pid, i, n_ego), end='\r')
                traj = online_adaption(env=env,
                                       policy=policy,
                                       max_steps=max_steps,
                                       obs=ground_truth['observations'][i, :, :],
                                       mean=ground_truth['actions'][i, :, :],
                                       env_kwargs=kwargs,
                                       lbd=lbd,
                                       adapt_steps=adapt_steps,
                                       output_dir=output_dir)
                traj_list.append(traj)

    return traj_list


def parallel_collect_trajectories(hyperparams=None,
                                  state=None,
                                  env_fn=None,
                                  policy_fn=None,
                                  egoids=None,
                                  starts=None,
                                  max_steps=200,
                                  use_hgail=False,
                                  n_proc=1,
                                  random_seed=None,
                                  lbd=0.99,
                                  adapt_steps=1,
                                  output_dir='',
                                  ground_truth=None):
    # build manager and dictionary mapping ego ids to list of trajectories
    manager = mp.Manager()
    trajlist = manager.list()
    # partition egoids 
    proc_egoids = utils.partition_list(egoids, n_proc)
    # pool of processes, each with a set of ego ids
    pool = mp.Pool(processes=n_proc)
    # run collection
    results = []
    for pid in range(n_proc):
        res = pool.apply_async(collect_trajectories,
                               args=(hyperparams,
                                     state,
                                     env_fn,
                                     policy_fn,
                                     egoids,
                                     starts,
                                     trajlist,
                                     proc_egoids[pid],
                                     max_steps,
                                     use_hgail,
                                     random_seed,
                                     lbd,
                                     adapt_steps,
                                     output_dir,
                                     ground_truth))
        results.append(res)

    # wait for the processes to finish
    [res.get() for res in results]
    pool.close()
    # let the julia processes finish up
    time.sleep(10)
    return trajlist


def single_process_collect_trajectories(hyperparams=None,
                                        state=None,
                                        env_fn=None,
                                        policy_fn=None,
                                        egoids=None,
                                        starts=None,
                                        max_steps=200,
                                        use_hgail=False,
                                        random_seed=None,
                                        output_dir='',
                                        ground_truth=None):
    # build list to be appended to 
    traj_list = []
    # collect trajectories in a single process
    collect_trajectories(hyperparams=hyperparams,
                         state=state,
                         env_fn=env_fn,
                         policy_fn=policy_fn,
                         egoids=egoids,
                         starts=starts,
                         traj_list=traj_list,
                         max_steps=max_steps,
                         use_hgail=use_hgail,
                         random_seed=random_seed,
                         output_dir=output_dir,
                         ground_truth=ground_truth)
    return traj_list


def load_egoids(filename, args, n_runs_per_ego_id=10, env_fn=None):
    offset = args.env_H + args.env_primesteps
    # TODO: pass params as arguments
    basedir = os.path.expanduser('~/.julia/packages/NGSIM/OPF1x/data/')
    ids_filename = filename.replace('.txt', '-index-{}-ids.h5'.format(offset))
    ids_filepath = os.path.join(basedir, ids_filename)
    if not os.path.exists(ids_filepath):
        # this should create the ids file
        env_fn(args)
        if not os.path.exists(ids_filepath):
            raise ValueError('file unable to be created, check args')
    ids = np.array(h5py.File(ids_filepath, 'r')['ids'].value)

    # we want to sample start times uniformly from the range of possible values 
    # but we also want these start times to be identical for every model we 
    # validate. So we sample the start times a single time, and save them.
    # if they exist, we load them in and reuse them
    start_times_filename = filename.replace('.txt', '-index-{}-starts.h5'.format(offset))
    start_times_filepath = os.path.join(basedir, start_times_filename)
    # check if start time filepath exists
    if os.path.exists(start_times_filepath):
        # load them in
        starts = np.array(h5py.File(start_times_filepath, 'r')['starts'].value)
    # otherwise, sample the start times and save them
    else:
        ids_file = h5py.File(ids_filepath, 'r')
        ts = ids_file['ts'].value
        # subtract offset gives valid end points
        te = ids_file['te'].value - offset
        starts = np.array([np.random.randint(s, e + 1) for (s, e) in zip(ts, te)])
        # write to file
        starts_file = h5py.File(start_times_filepath, 'w')
        starts_file.create_dataset('starts', data=starts)
        starts_file.close()

    # create a dict from id to start time
    id2starts = dict()
    for (egoid, start) in zip(ids, starts):
        id2starts[egoid] = start

    ids = np.tile(ids, n_runs_per_ego_id)
    return ids, id2starts


def main(options):
    if not options.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 1: INFO, 2: WARNING, 3: ERROR
    if options.data_dir == '':
        raise FileNotFoundError('Empty dataset dir')
    dataset_dir = os.path.realpath(os.path.join(os.path.expanduser(options.data_dir), 'dataset', options.dataset))
    feature_database_path = os.path.join(dataset_dir, options.feature_database)
    pretrained_dir = os.path.realpath(os.path.join(options.data_dir, 'pretrained'))
    validation_dir = os.path.realpath(os.path.join(options.data_dir, 'validation'))
    utils.maybe_mkdir(validation_dir)

    # Hyperparams
    hyperparams_path = os.path.join(pretrained_dir, 'args.npz')
    hyperparams = hp.load_args(os.path.join(pretrained_dir, hyperparams_path))
    if options.use_multiagent:
        hyperparams.env_multiagent = True
        hyperparams.remove_ngsim_vehicles = options.remove_ngsim_vehicles
    if options.n_envs:
        hyperparams.n_envs = options.n_envs
    print('{} vehicles with H = {}'.format(hyperparams.n_envs, hyperparams.env_H)) # args.env_H should be 200

    # pretrained state
    state_path = os.path.join(pretrained_dir, options.state)
    state = hgail.misc.utils.load_params(state_path)

    # Validation set
    # TODO: check what this does
    if len(options.subsets) == 0:
        # TODO: select subsets of current section only
        subset_file_names = [fn for fn in os.listdir(dataset_dir) if '.txt' in fn and 'trajdata' in fn]
    else:
        subset_file_names = options.subsets
    print('Found {} data subsets:'.format(len(subset_file_names)))
    utils.print_list(subset_file_names)

    # ground_truth: a dict of the following keys:
    # actions, shape (22, 331, 2)
    # observations, shape (22, 331, 66)
    # obs_mean, shape (1, 66)
    # obs_std, shape (1, 66)
    ground_truth = validate_utils.get_ground_truth(feature_database_path=feature_database_path,
                                                   traj_data_file_name=subset_file_names)

    if options.debug:
        collect_fn = single_process_collect_trajectories
    else:
        collect_fn = parallel_collect_trajectories

    for idx_subset, subset in enumerate(subset_file_names):
        print('Training {}/{} subsets: {}...'.format(idx_subset + 1, len(subset_file_names), subset))
        subset_name = os.path.splitext(subset)[0]
        hyperparams.ngsim_filename = subset
        output_traj_path = os.path.join(validation_dir, subset_name, '{}_AGen.npz'.format(subset_name))
        if hyperparams.env_multiagent:
            # args.n_envs gives the number of simultaneous vehicles
            # so run_args.n_multiagent_trajs / args.n_envs gives the number
            # of simulations to run overall
            egoids = list(range(int(options.n_multiagent_trajs / hyperparams.n_envs)))
            starts = dict()
        else:
            egoids, starts = load_egoids(subset, hyperparams, options.n_runs_per_ego_id)

        # get policy
        policy_fn = utils.build_hierarchy if options.use_hgail else validate_utils.build_policy
        if not options.use_multiagent:
            tf.reset_default_graph()

        subset_output_dir = os.path.join(validation_dir, subset_name)
        utils.maybe_mkdir(subset_output_dir)

        trajs = collect_fn(hyperparams=hyperparams,
                           state=state,
                           egoids=egoids,
                           starts=starts,
                           env_fn=utils.build_ngsim_env,  # TODO: two build_ngsim_env in utils.py and validate_utils.py, check which one to use
                           policy_fn=policy_fn,
                           max_steps=options.max_steps,
                           use_hgail=options.use_hgail,
                           n_proc=options.n_proc,
                           random_seed=options.random_seed,
                           lbd=options.lbd,
                           adapt_steps=options.adapt_steps,
                           output_dir=subset_output_dir,
                           ground_truth=ground_truth)
        utils.write_trajectories(output_traj_path, trajs)
    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptation options')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='../../data/experiments/agen/adaption')
    parser.add_argument('--dataset', type=str, default='multi_agent_22')
    parser.add_argument('--feature_database', type=str, default='ngsim_22agents.h5')
    parser.add_argument('--subsets', nargs='+', default=['trajdata_i101_trajectories-0750am-0805am.txt'])
    parser.add_argument('--state', type=str, default='itr_2000.npz')
    parser.add_argument('--n_runs_per_ego_id', type=int, default=10)
    parser.add_argument('--use_hgail', action='store_true', default=False)
    parser.add_argument('--use_multiagent', action='store_true', default=False)
    parser.add_argument('--n_multiagent_trajs', type=int, default=10000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--remove_ngsim_vehicles', action='store_true', default=False)
    parser.add_argument('--lbd', type=float, default=0.99)
    parser.add_argument('--adapt_steps', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=False)

    options = parser.parse_args()

    main(options)
