import argparse
import logging
import os, sys
import csv
import numpy as np
import random
import time

from run_ple_utils import make_ple_env

def main_event_dependent():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_env', help='testv environment ID', default='ContFlappyBird-v3')
    parser.add_argument('--total_timesteps', help='Total number of env steps', type=int, default=int(2e5))
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--logdir', default='/home/mara/Desktop/logs/ED_CONTROL',
                        help='directory where logs are stored')
    parser.add_argument('--show_interval', type=int, default=1,
                        help='Env is rendered every n-th episode. 0 = no rendering')
    parser.add_argument('--eval_model', choices=['all', 'inter', 'final'], default='inter',
                        help='Eval all stored models, only the final model or only the intermediately stored models (while testing the best algorithm configs)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Init test_results.csv
    # rnd_output_dir = args.logdir
    #
    # logger = logging.getLogger()
    # fh = logging.FileHandler(os.path.join(rnd_output_dir, 'algo.log'))
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    # logger.addHandler(fh)
    # logger.setLevel(logging.INFO)
    # logger.propagate = False
    #
    # result_path = os.path.join(rnd_output_dir, 'test_results.csv')

    for s in range(100, 120):
        # logger.info('make env with seed %s' % s)
        test_env = make_ple_env(args.test_env, seed=s)

        state = test_env.reset()
        #print(state)
        # logger.info('reset')
        total_return = 0
        rew_traj =[]

        t = 0
        while t < args.total_timesteps:
            t+=1
            if t%20 == 0:
                a = 1
            if args.show_interval > 0:
                test_env.render()
                time.sleep(0.01)
                # logger.info('render')
            # logger.info('step')

            if state[0] > 0.5*(state[2]+state[3]):
                action = 0  # FLAP
            else:
                action = 1
            state, reward, dones, _ = test_env.step(action)
            #print(state)
            # logger.info('stepped')
            # reward_window.append(reward)
            total_return += reward
            rew_traj.append(reward)
        test_env.close()

        # with open(result_path, "a") as csvfile:
        #     logger.info('write csv')
        #     writer = csv.writer(csvfile)
        #     # rew_traj[0:0] = [s, 0, np.mean(rew_traj)]
        #     # writer.writerow(rew_traj)
        #     writer.writerow([s, 0, np.mean(rew_traj)])


if __name__ == '__main__':
    main_event_dependent()