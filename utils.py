import os
import random
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
import multiprocessing
import logging

logger = logging.getLogger(__name__)


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def make_session(num_cpu=None, make_default=False):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.allocator_type = 'BFC'
    if make_default:
        return tf.InteractiveSession(config=tf_config)
    else:
        return tf.Session(config=tf_config)

# ---------------------------------------------------------------------------------------
#                           model utils
# ---------------------------------------------------------------------------------------


class CategoricalPd:
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        # add small unifrom noise to distribution --> further exploration
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_actions)


def sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


def stable_softmax(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)  # shift values
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return p0


def cat_entropy(logits):
    # stable_softmax using logits as inputs
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)  # shift values
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0  # softmax output
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def cat_entropy_softmax(p0):
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)


def mse(pred, target):
    return tf.square(pred-target)/2.


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value # second dimension of input matrix (length of feature/observation vector)
        w = tf.get_variable("w", [nin, nh], initializer=tf.orthogonal_initializer(gain=init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(x, w)+b


def lstm(xs, ms, s, scope, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    nsteps = len(xs)
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(init_scale))
        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s


def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x


# -----------------------------------------------------------------------------
#                                   RL utils
# -----------------------------------------------------------------------------

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


def normalize_obs(obs):
    """
    Normalizes values of observation features of ContinuousFlappyBird environment to values between 0 and 1.

    Parameters
    ----------
    obs : vector of observations

    Returns
    -------
    normalized vector with values between 0 and 1.
    """
    scaling_values = [512, 7, 512, 512, 512, 512, 512, 512]

    # Observations from a single environment
    if len(obs.shape) == 1:
        obs_len = len(obs)
        if obs_len > 8:  # (nfeat,)
            n_randfeat = obs_len - 8
            for i in range(n_randfeat):
                scaling_values.append(512)
        obs = [obs[j] / scaling_values[j] for j in range(obs_len)]

    else:  # Obs from multiple parallel envs
        obs_len = len(obs[0])
        if obs_len > 8:  # (nfeat,)
            n_randfeat = obs_len - 8
            for i in range(n_randfeat):
                scaling_values.append(512)

        for i in range(len(obs)):
            obs[i] = [obs[i,j] / scaling_values[j] for j in range(obs_len)]
    return np.asarray(obs)


def make_path(f):
    return os.makedirs(f, exist_ok=True)


# These functions allows us to update the parameters of our target network with those of the primary network.
def update_target_graph(train_vars,tau):
    nvars = len(train_vars)
    op_holder = []
    for idx, var in enumerate(train_vars[0:nvars//2]):
        new_var = var.value()  # values of train_network
        target_var = train_vars[idx+nvars//2].value()  # values of target network
        op_holder.append(train_vars[idx+nvars//2].assign((tau*new_var) + ((1-tau)*target_var)))
    return op_holder

# -----------------------------------------------------------------------------
#                              DQN - specific utils
# -----------------------------------------------------------------------------


def make_epsilon_greedy_policy(predict_fn, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        predict_fn: An estimator function that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation and epsilon (The probability to select a random action [0,1]) as an
        argument and returns the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = predict_fn(observation, None, None)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, max_replay_buffer_size):
        self.max_replay_buffer_size = max_replay_buffer_size
        self._data = namedtuple("ReplayBuffer", ["obs", "actions", "next_obs", "rewards", "values", "dones"])  #, "states"])
        self._data = self._data(obs=[], actions=[], next_obs=[], rewards=[], values=[], dones=[])  #, states=[])

    def add_transition(self, obs, action, next_obs, reward, done, values=None):
        if self.size() > self.max_replay_buffer_size:
            self._data.obs.pop(0)
            self._data.actions.pop(0)
            self._data.next_obs.pop(0)
            self._data.rewards.pop(0)
            self._data.dones.pop(0)
            self._data.values.pop(0)
        self._data.obs.append(obs)
        self._data.actions.append(action)
        self._data.next_obs.append(next_obs)
        self._data.rewards.append(reward)
        self._data.dones.append(done)
        self._data.values.append(values)

    def next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.obs), batch_size)  # TODO prioritized experience replay
        batch_obs = np.array([self._data.obs[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_obs = np.array([self._data.next_obs[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_values = np.array([self._data.values[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_values, batch_dones  #, batch_states

    def recent_and_next_batch(self, batch_size):
        # Draw batch_sze/2 random samples
        n_batched = batch_size//2
        batch_indices = np.random.choice(len(self._data.obs), n_batched)  # TODO prioritized experience replay
        batch_obs = np.array([self._data.obs[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_obs = np.array([self._data.next_obs[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_values = np.array([self._data.values[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])

        # Take batch_size/2 recent samples and add them to mini batch
        buffer_length = self.size()
        n_recent = batch_size - n_batched
        batch_obs = np.concatenate((batch_obs, self._data.obs[buffer_length-n_recent:]))
        batch_actions = np.concatenate((batch_actions, self._data.actions[buffer_length-n_recent:]))
        batch_next_obs = np.concatenate((batch_next_obs, self._data.next_obs[buffer_length-n_recent:]))
        batch_rewards = np.concatenate((batch_rewards, self._data.rewards[buffer_length-n_recent:]))
        batch_values = np.concatenate((batch_values, self._data.values[buffer_length-n_recent:]))
        batch_dones = np.concatenate((batch_dones, self._data.dones[buffer_length-n_recent:]))

        return batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_values, batch_dones  #, batch_states

    def recent_and_next_batch_of_seq(self, batch_size, trace_length):
        # Draw batch_sze/2 random samples
        n_batched = batch_size-1

        batch_indices = np.random.choice(len(self._data.obs) - trace_length, n_batched)  # TODO len - trace_length prioritized experience replay
        batch_obs =[self._data.obs[i:i + trace_length] for i in batch_indices]
        batch_actions = [self._data.actions[i:i + trace_length] for i in batch_indices]
        batch_next_obs = [self._data.next_obs[i:i + trace_length] for i in batch_indices]
        batch_rewards = [self._data.rewards[i:i + trace_length] for i in batch_indices]
        batch_values = [self._data.values[i:i + trace_length] for i in batch_indices]
        batch_dones = [self._data.dones[i:i + trace_length] for i in batch_indices]

        # Take batch_size/2 recent samples and add them to mini batch
        buffer_length = self.size()
        batch_obs.append(self._data.obs[buffer_length - trace_length:])
        batch_actions.append(self._data.actions[buffer_length - trace_length:])
        batch_next_obs.append(self._data.next_obs[buffer_length - trace_length:])
        batch_rewards.append(self._data.rewards[buffer_length - trace_length:])
        batch_values.append(self._data.values[buffer_length - trace_length:])
        batch_dones.append(self._data.dones[buffer_length - trace_length:])

        batch_obs = np.reshape(np.array(batch_obs), newshape=[batch_size * trace_length, -1])
        batch_actions = np.reshape(np.array(batch_actions), newshape=[batch_size * trace_length])
        batch_next_obs = np.reshape(np.array(batch_next_obs), newshape=[batch_size * trace_length, -1])
        batch_rewards = np.reshape(np.array(batch_rewards), newshape=[batch_size * trace_length])
        batch_values = np.reshape(np.array(batch_values), newshape=[batch_size * trace_length])
        batch_dones = np.reshape(np.array(batch_dones), newshape=[batch_size * trace_length])

        # batch_states = np.array([self._data.states[i] for i in batch_indices]).squeeze()
        return batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_values, batch_dones  #, batch_states

    def size(self):
        return len(self._data.obs)

# def batch_to_seq(h, nbatch, nsteps, flat=False):
#     if flat:
#         h = tf.reshape(h, [nbatch, nsteps])
#     else:
#         h = tf.reshape(h, [nbatch, nsteps, -1])
#     return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]
#
#
# def seq_to_batch(h, flat = False):
#     shape = h[0].get_shape().as_list()
#     if not flat:
#         assert(len(shape) > 1)
#         nh = h[0].get_shape()[-1].value
#         return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
#     else:
#         return tf.reshape(tf.stack(values=h, axis=1), [-1])

# -----------------------------------------------------------------------------
#                        Episode Analysis
# -----------------------------------------------------------------------------
class EpisodeStats:
    def __init__(self, nsteps, nenvs):
        self.episode_rewards = []
        for i in range(nenvs):
            self.episode_rewards.append([])
        self.lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
        self.nsteps = nsteps
        self.nenvs = nenvs

    def feed(self, rewards, masks):
        rewards = np.reshape(rewards, [self.nenvs, self.nsteps])
        masks = np.reshape(masks, [self.nenvs, self.nsteps])
        for i in range(0, self.nenvs):
            for j in range(0, self.nsteps):
                self.episode_rewards[i].append(rewards[i][j])
                if masks[i][j]:
                    l = len(self.episode_rewards[i])
                    s = sum(self.episode_rewards[i])
                    self.lenbuffer.append(l)
                    self.rewbuffer.append(s)
                    self.episode_rewards[i] = []

    def mean_length(self):
        if self.lenbuffer:
            return np.mean(self.lenbuffer)
        else:
            return 0  # on the first params dump, no episodes are finished

    def mean_reward(self):
        if self.rewbuffer:
            return np.mean(self.rewbuffer)
        else:
            return 0

    def total_return_env(self, env_idx):
        return


def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

# --------------------------------------------------------------------------------
#                          Save & Restore models
# --------------------------------------------------------------------------------

def add_to_collection_rnn_state(name, rnn_state):
    """
    Add the recurrent network state to the collection which is stored.
    - From kaufmanu on stackoverflow
    """
    # store the name of each cell type in a different collection
    coll_of_names = name + '__names__'
    for layer in rnn_state:
        n = layer.__class__.__name__
        tf.add_to_collection(coll_of_names, n)
        try:
            for l in layer:
                tf.add_to_collection(name, l)
        except TypeError:
            # layer is not iterable so just add it directly
            tf.add_to_collection(name, layer)


def get_collection_rnn_state(name):
    """
    Get the recurrent network state from the stored model.
    - From kaufmanu on stackoverflow
    """
    layers = []
    coll = tf.get_collection(name)
    coll_of_names = tf.get_collection(name + '__names__')
    idx = 0
    for n in coll_of_names:
        if 'LSTMStateTuple' in n:
            state = tf.nn.rnn_cell.LSTMStateTuple(coll[idx], coll[idx+1])
            idx += 2
        else:  # add more cell types here
            state = coll[idx]
            idx += 1
        layers.append(state)
    return tuple(layers)

# def add_to_collection_rnn_state(name, rnn_state):
#     # store the name of each cell type in a different collection
#     coll_of_names = name + '__names__'
#     try:
#         for layer in rnn_state:
#             n = layer.__class__.__name__
#             tf.add_to_collection(coll_of_names, n)
#             try:
#                 for l in layer:
#                     tf.add_to_collection(name, l)
#             except TypeError:
#                 # layer is not iterable so just add it directly
#                 tf.add_to_collection(name, layer)
#     except TypeError:
#         # layer is not iterable so just add it directly
#         tf.add_to_collection(name, rnn_state)
#
#
# def get_collection_rnn_state(name):
#     layers = []
#     coll = tf.get_collection(name)
#     coll_of_names = tf.get_collection(name + '__names__')
#     idx = 0
#     if coll_of_names == []:
#         layers.append(coll[0])
#     else:
#         for n in coll_of_names:
#             try:
#                 if 'LSTMStateTuple' in n:
#                     state = tf.nn.rnn_cell.LSTMStateTuple(coll[idx], coll[idx+1])
#                     idx += 2
#                 else:  # add more cell types here
#                     state = coll[idx]
#                     idx += 1
#             except TypeError:
#                 # else:  # add more cell types here
#                 state = coll[idx]
#                 idx += 1
#             layers.append(state)
#         return tuple(layers)


def reset_model(train_vars, train_vars_values):
    op_holder=[]
    for var, val in zip(train_vars, train_vars_values):
        op_holder.append(var.assign(val))
    return op_holder