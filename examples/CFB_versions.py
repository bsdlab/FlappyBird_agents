"""
This module gives examples on how to use the implemented environment ContinuousFlappyBird (CFB).

Refer to envs/ple/games/__init__.py for details of implementation of the baseline CFB.
Refer to envs/nenvironment.py to learn how non-stationarities, additive noise on the state representation and additional
random features are implemented.
Refer to the module envs/random_trajectories.py to learn more about how sequences of model parameters and random
features are pre-generated.

"""

import numpy as np
from run_ple_utils import make_ple_env, make_ple_envs

RENDER = True


def main():
    seed = 15

    # ---- Specifiy the version of CFB ----
    game = 'ContFlappyBird'
    ns = 'gfNS'                         # 'gfNS', 'gsNS', 'rand_feat'
    nrandfeat = ('-nrf' + str(2))       # '', 0,2,3,4
    noiselevel = ('-nl' + str(0.001))   # '', 0.0001 - 0.05 (see env/__init__.py)
    experiment_phase = '-test'          # '-test', '-train'

    # Naming convention is <game>-<non-stationarity>-nl<noise_level>-nrf<nrandfeat>-<phase>-v0
    env_name = (game + '-' + ns + noiselevel + nrandfeat + experiment_phase + '-v0')

    # ---- Generate CFB with single instance ----
    env = make_ple_env(env_name, seed=seed)
    # Run env:
    env.seed(seed=seed)
    env.reset()
    for i in range(100):
        state, reward, done, info = env.step(action=np.random.randint(len(env.action_space)+1))
        if RENDER:
            env.render()

    # ---- Generate CFB with N parallel instances. ----
    N = 3
    env = make_ple_envs(env_name, num_env=N, seed=seed)
    # Run env:
    env.seed(seed=seed)
    env.reset()
    for i in range(100):
        state, reward, done, info = env.step(action=np.random.randint(len(env.action_space)+1))
        if RENDER:
            env[0].render()


if __name__ == '__main__':
    main()
