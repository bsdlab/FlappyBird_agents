import os, glob
import csv
import logging

import tensorflow as tf
import numpy as np
import time
from utils import set_global_seeds, normalize_obs, get_collection_rnn_state
from run_ple_utils import make_ple_env

SEED = 100
LOGDIR = '/home/mara/Videos'
F_NAME = 'final_model-2000000'

ple_env = make_ple_env('ContFlappyBird-v3', seed=SEED)
tf.reset_default_graph()
set_global_seeds(SEED)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # g = tf.get_default_graph()  # Shouldn't be set here again, as a new RNG is used without previous seeding.

    # restore the model
    loader = tf.train.import_meta_graph(glob.glob(os.path.join(LOGDIR, (F_NAME + '.meta')))[0])

    # now variables exist, but the values are not initialized yet.
    loader.restore(sess, os.path.join(LOGDIR, F_NAME))  # restore values of the variables.

    # Load operations from collections
    obs_in = tf.get_collection('inputs')
    probs_out = tf.get_collection('pi')
    pi_logits_out = tf.get_collection('pi_logit')
    predict_vf_op = tf.get_collection('val')
    predict_ac_op = tf.get_collection('step')
    rnn_state_in, rnn_state_out = None, None

    env = ple_env
    pi_out = probs_out

    logger = logging.getLogger(__name__)
    ep_length = []
    ep_return = []
    logger.info('---------------- Episode results -----------------------')
    for i in range(0,
                   2):  # TODO parallelize this here! Problem: guarantee same sequence of random numbers in each parallel process. --> Solution Index based RNG instead of sequential seed based RNG
        obs = env.reset()
        obs = normalize_obs(obs)
        done = False

        i_sample = 0

        while not done and (i_sample < 5000):
            i_sample += 1
            pi, pi_log, act = sess.run([pi_out, pi_logits_out, predict_ac_op], feed_dict={obs_in[0]: [obs]})
            ac = np.argmax(pi_log)
            obs, reward, done, _ = env.step(ac)
            # obs, reward, done, _ = env.step(act[0][0])
            obs = normalize_obs(obs)

            env.render()
            time.sleep(0.01)
