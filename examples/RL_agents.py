"""
This module gives examples on how to learn an optimal policy using the implemented RL methods in ContinuousFlappyBird (CFB).
"""

from models import MLPPolicy, LSTMPolicy, GRUPolicy
from run_ple_utils import make_ple_envs, make_ple_env

LOGDIR = '/home/mara/Desktop'  # TODO set the directory

def main():
    seed = 42

    # ---- Specifiy the version of CFB ----
    game = 'ContFlappyBird'
    ns = ''                             # '', 'gfNS', 'gsNS', 'rand_feat'
    nrandfeat = ('-nrf' + str(2))       # '', 0,2,3,4
    noiselevel = ('-nl' + str(0.001))   # '', 0.0001 - 0.05 (see env/__init__.py)
    experiment_phase = '-train'         # '-test', '-train'

    # Naming convention is <game>-<non-stationarity>-nl<noise_level>-nrf<nrandfeat>-<phase>-v0
    env_name = (game + ns + noiselevel + nrandfeat + experiment_phase + '-v0')
    test_env_name = (game + ns + noiselevel + nrandfeat + '-test' + '-v0')

    # ---- Generate CFB with N parallel instances and with single instance ----
    ple_env = make_ple_envs(env_name, num_env=2, seed=seed)  # N parallel instances
    test_env = make_ple_env(test_env_name, seed=seed+42)  # single instance

    # ---- Import the RL method you want to use ----
    from A2C.a2c import learn
    # from PPO.ppo import learn
    # from DQN.dqn import q_learning

    # ---- Specify the model (FF, LSTM, GRU) ----
    model_architecture = 'ff'  # 'lstm', 'gru'

    if model_architecture == 'ff':
        policy_fn = MLPPolicy
    elif model_architecture == 'lstm':
        policy_fn = LSTMPolicy
    elif model_architecture == 'gru':
        policy_fn = GRUPolicy
    else:
        print('Policy option %s is not implemented yet.' % model_architecture)

    # ---- Learn an optimal policy. The agents model ('final_model...') is stored in LOGDIR.
    early_stopped = learn(policy_fn,
                          env=ple_env,
                          test_env=test_env,
                          seed=seed,
                          total_timesteps=int(2e4),  # Total number of env steps
                          log_interval=0,  # Network parameter values are stored in tensorboard summary every <log_interval> model update step. 0 --> no logging
                          test_interval=0,  # Model is evaluated after <test_interval> model updates. 0 = do not evaluate while learning.
                          show_interval=0,  # Env is rendered every n-th episode. 0 = no rendering
                          logdir=LOGDIR,  # directory where logs and the learned models are stored
                          lr=5e-4,  # Learning Rate
                          max_grad_norm=0.01,  # Maximum gradient norm up to which gradient is not clipped
                          units_per_hlayer=(64, 64, 64),  # Number of units per network layer
                          activ_fcn='relu6',  # Type of activation function used in the network: 'relu6', 'elu', 'mixed'
                          gamma=0.95,  # Discount factor for discounting the reward
                          vf_coef=0.2,  # Weight on the value function loss in the loss function
                          ent_coef=1e-7,   # Weight on the policy entropy in the loss function
                          batch_size=64,  # number of samples based on which gradient is updated
                          early_stop=False,  # whether or not to stop bad performing runs earlier.
                          keep_model=0)  # How many best models shall be kept during training. 0 -> only final model
    ple_env.close()


if __name__ == '__main__':
    main()
