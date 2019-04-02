#!/usr/local/bin/python3.6

import sys, os
import logging

print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))
from run_ple_utils import make_ple_envs, make_ple_env, arg_parser, params_parser
from models import MLPPolicy, LSTMPolicy, GRUPolicy
from PPO.eval_ppo_model import eval_model
from PPO.ppo import learn


# Run this function in SMAC script. It takes the arguments from the function call and sets unset
# arguments to their default value.


def ppo_params_parser(**kwargs):
    param_dict = params_parser()
    param_dict.add_num_param("nenvs", lb=1, ub=30, default=3, dtype=int)
    param_dict.add_num_param("vf_coeff", lb=1e-2, ub=1., default=0.2, dtype=float)
    param_dict.add_num_param("ent_coeff", lb=1e-12, ub=1., default=1e-7, dtype=float)
    param_dict.add_num_param("units_shared_layer1", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("units_shared_layer2", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("units_policy_layer", lb=1, ub=700, default=64, dtype=int)
    param_dict.add_num_param("nminibatches", lb=1, ub=500, default=1, dtype=int)
    param_dict.add_num_param("noptepochs", lb=1, ub=500, default=1, dtype=int)
    param_dict.add_num_param("lam", lb=0, ub=1., default=0.95, dtype=float)
    param_dict.add_num_param("nsteps", lb=1, ub=500, default=32, dtype=int)
    param_dict.add_num_param("cliprange", lb=0., ub=1., default=0.2, dtype=float)

    return param_dict.check_params(**kwargs)


def run_ppo_smac(**kwargs):
    params = ppo_params_parser(**kwargs)

    seed = params["seed"]
    ple_env = make_ple_envs(params["env"], num_env=params["nenvs"], seed=seed)
    test_env = make_ple_env(params["test_env"], seed=3000)

    if params["architecture"] == 'ff':
        policy_fn = MLPPolicy
    elif params["architecture"] == 'lstm':
        policy_fn = LSTMPolicy
    elif params["architecture"] == 'gru':
        policy_fn = GRUPolicy
    else:
        print('Policy option %s is not implemented yet.' % params["policy"])

    with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
        for k, v in params.items():
            f.write(k + ': ' + str(v) + '\n')
    print(params)

    early_stopped = learn(policy_fn,
                          env=ple_env,
                          test_env=test_env,
                          seed=seed,
                          total_timesteps=params["total_timesteps"],
                          log_interval=params["log_interval"],
                          test_interval=params["test_interval"],
                          show_interval=params["show_interval"],
                          logdir=params["logdir"],
                          lr=params["lr"],
                          # lrschedule=params["lrschedule"],
                          max_grad_norm=params["max_grad_norm"],
                          units_per_hlayer=(params["units_shared_layer1"],
                                            params["units_shared_layer2"],
                                            params["units_policy_layer"]),
                          activ_fcn=params["activ_fcn"],
                          gamma=params["gamma"],
                          vf_coef=params["vf_coeff"],
                          ent_coef=params["ent_coeff"],
                          nsteps=params["nsteps"],
                          lam=params["lam"],
                          nminibatches=params["nminibatches"],
                          noptepochs=params["noptepochs"],
                          cliprange=params["cliprange"],
                          early_stop=params["early_stop"],
                          keep_model=params["keep_model"])
    ple_env.close()

    if not early_stopped:
        avg_perf, var_perf, max_return = eval_model(render=False, nepisodes=10, test_steps=3000, **params)

        with open(os.path.join(params["logdir"], 'hyperparams.txt'), 'a') as f:
            f.write('\n')
            f.write('Results: \n')
            f.write('average performance: ' + str(avg_perf) + '\n')
            f.write('performance variance: ' + str(var_perf) + '\n')
            f.write('maximum return: ' + str(max_return) + '\n')

        return avg_perf, var_perf, max_return
    else:
        return -3000, 3000, -3000


def main():
    parser = arg_parser()
    parser.add_argument('--early_stop', help='stop bad performing runs ealier', type=bool, default=False)
    parser.add_argument('--nenvs', help='Number of parallel simulation environmenrs', type=int, default=4)
    parser.add_argument('--activ_fcn', choices=['relu6', 'elu', 'mixed'], type=str, default='elu',
                        help='Activation functions of network layers', )
    parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--nsteps', type=int, default=32, help='number of samples based on which gradient is updated')
    parser.add_argument('--nminibatches', help='Number of minibatches per sampled data batch.', type=int, default=1)
    parser.add_argument('--noptepochs', help='Number of optimization epochs with sample data, i.e. how often samples are reused.', type=int, default=1)

    parser.add_argument('--lam', help='Lambda parameter for GAE', type=float, default=0.95)
    parser.add_argument('--cliprange', help='Defines the maximum policy change allowed, before clipping.', type=float, default=0.2)
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.90)
    parser.add_argument('--vf_coeff', help='Weight of value function loss in total loss', type=float, default=0.2)
    parser.add_argument('--ent_coeff', help='Weight of entropy in total loss', type=float, default=1e-7)
    parser.add_argument('--units_shared_layer1', help='Units in first hidden layer which is shared', type=int, default=64)
    parser.add_argument('--units_shared_layer2', help='Units in second hidden layer which is shared', type=int, default=64)
    parser.add_argument('--units_policy_layer', help='Units in hidden layer in policy head', type=int, default=64)

    parser.add_argument('--restore_model', help='whether a pretrained model shall be restored', type=bool, default=False)
    args = parser.parse_args()

    seed = args.seed
    env = make_ple_envs(args.env, num_env=args.nenvs, seed=seed*10)
    # env = make_ple_envs('ContFlappyBird-hNS-nrf2-train-v0', num_env=args.nenvs, seed=seed - 1)
    test_env = make_ple_env(args.test_env, seed=3000)

    if args.architecture == 'ff':
        policy_fn = MLPPolicy
    elif args.architecture == 'lstm':
        policy_fn = LSTMPolicy
    elif args.architecture == 'gru':
        policy_fn = GRUPolicy
    else:
        print('Policy option %s is not implemented yet.' % args.policy)

    # store hyperparms setting
    # logdir = os.path.join(args.logdir, str(datetime.datetime.today()))
    # os.makedirs(logdir)

    ppo_output_dir = os.path.join(args.logdir, ('ppo_output'+str(args.seed)))
    if not os.path.isdir(ppo_output_dir):
        os.makedirs(ppo_output_dir)

    with open(os.path.join(ppo_output_dir, 'hyperparams.txt'), 'a') as f:
        for k,v in vars(args).items():
            f.write(k + ': ' + str(v) + '\n')

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(ppo_output_dir, 'algo.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    early_stopped = learn(policy_fn,
                          env=env,
                          test_env=test_env,
                          seed=seed,
                          total_timesteps=args.total_timesteps,
                          log_interval=args.log_interval,
                          test_interval=args.test_interval,
                          show_interval=args.show_interval,
                          logdir=ppo_output_dir,
                          lr=args.lr,
                          # lrschedule=args.lrschedule,
                          max_grad_norm=args.max_grad_norm,
                          units_per_hlayer=(args.units_shared_layer1,
                                            args.units_shared_layer2,
                                            args.units_policy_layer),
                          activ_fcn=args.activ_fcn,
                          gamma=args.gamma,
                          vf_coef=args.vf_coeff,
                          ent_coef=args.ent_coeff,
                          nsteps=args.nsteps,
                          lam=args.lam,
                          nminibatches=args.nminibatches,
                          noptepochs=args.noptepochs,
                          cliprange=args.cliprange,
                          early_stop=args.early_stop,
                          keep_model=args.keep_model,
                          restore_model=args.restore_model)
    env.close()


if __name__ == '__main__':
    main()
