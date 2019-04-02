import numpy as np
import tensorflow as tf
from utils import CategoricalPd, fc


def fc_layers(game_state):  # TODO decide on architecture
    activ = tf.tanh
    h = activ(fc(game_state, 'fc1', nh=64))
    h1 = activ(fc(h, 'fc2', nh=64))
    return activ(fc(h1, 'fc3', nh=32))


def random_choice(sess, data, probs):
    data = tf.convert_to_tensor(data)
    assert data.shape == probs.shape, 'array and probability need to have the same shape'
    idx_sample = tf.multinomial(tf.log(probs), 1)
    return data[tf.cast(idx_sample[0][0], tf.int32)].eval(session=sess)


# -------------------------------------------------------------------------------------------------
#                                           A2C and PPO
# -------------------------------------------------------------------------------------------------
class MLPPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenvs, nsteps, units_per_hlayer, reuse=False, activ_fcn='relu6'):  # pylint: disable=W0613
        # this method is called with nbatch = nenvs*nsteps

        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc)
        # actdim = ac_space.shape[0]
        # Todo check initialization
        # Input and Output dimensions
        nd, = ob_space.shape
        nbatch = nenvs * nsteps
        ob_shape = (nbatch, nd)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            if activ_fcn == 'relu6':
                h1 = tf.nn.relu6(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  # , init_scale=np.sqrt(2)))
                h2 = tf.nn.relu6(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  # , init_scale=np.sqrt(2)))

                h3 = tf.nn.relu6(fc(h2, 'pi_fc1', nh=units_per_hlayer[2]))  # , init_scale=np.sqrt(2)))
            elif activ_fcn == 'elu':
                h1 = tf.nn.elu(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  # , init_scale=np.sqrt(2)))
                h2 = tf.nn.elu(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  # , init_scale=np.sqrt(2)))

                h3 = tf.nn.elu(fc(h2, 'pi_fc1', nh=units_per_hlayer[2]))  # , init_scale=np.sqrt(2)))
            elif activ_fcn == 'mixed':
                h1 = tf.nn.relu6(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  #, init_scale=np.sqrt(2)))
                h2 = tf.nn.relu6(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  #, init_scale=np.sqrt(2)))

                h3 = tf.nn.tanh(fc(h2, 'pi_fc1', nh=units_per_hlayer[2]))  #, init_scale=np.sqrt(2)))

            pi_logit = fc(h3, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logit)

            vf = fc(h2, 'vf', 1)[:, 0]  # predicted value of input state

        self.pd = CategoricalPd(pi_logit)  # pdparam
        a0 = self.pd.sample()  # returns action index: 0,1
        # a0 = tf.argmax(pi, axis=1)
        neglogp0 = self.pd.neglogp(a0)

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, pi, v, neglogp = sess.run([a0, pi_logit, vf, neglogp0], {X: ob})
            return a, pi, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.pi_logit = pi_logit
        self.vf = vf
        self.ac = a0
        self.step = step
        self.value = value

class LSTMPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenvs, nsteps, units_per_hlayer, reuse=False, activ_fcn='relu6'):  # pylint: disable=W0613
        # this method is called with nbatch = nenvs*nsteps

        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc)
        # actdim = ac_space.shape[0]
        # Todo check initialization
        # Input and Output dimensions
        nd, = ob_space.shape
        nbatch = nenvs * nsteps
        ob_shape = (nbatch, nd)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            if activ_fcn == 'relu6':
                h1 = tf.nn.relu6(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  # , init_scale=np.sqrt(2)))
                h2 = tf.nn.relu6(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  # , init_scale=np.sqrt(2)))
            elif activ_fcn == 'elu':
                h1 = tf.nn.elu(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  # , init_scale=np.sqrt(2)))
                h2 = tf.nn.elu(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  # , init_scale=np.sqrt(2)))
            elif activ_fcn == 'mixed':
                h1 = tf.nn.relu6(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  #, init_scale=np.sqrt(2)))
                h2 = tf.nn.relu6(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  #, init_scale=np.sqrt(2)))

            # The output matrix [nbatch x trace_length, h_units] of layer 2 needs to be reshaped to a vector with
            # dimensions: [nbatch , trace_length , h_units] for rnn processing.
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=units_per_hlayer[1], state_is_tuple=True)
            rnn_input = tf.reshape(h2, shape=[nenvs, nsteps, units_per_hlayer[1]])
            rnn_state_in = rnn_cell.zero_state(batch_size=nenvs,
                                               dtype=tf.float32)  # reset the state in every training iteration
            rnn_output, rnn_state_out = tf.nn.dynamic_rnn(inputs=rnn_input,
                                                          cell=rnn_cell,
                                                          initial_state=rnn_state_in,
                                                          dtype=tf.float32,
                                                          scope="model" + '_rnn')
            # The output of the recurrent cell then needs to be reshaped to the original matrix shape.
            rnn_output = tf.reshape(rnn_output, shape=[-1, units_per_hlayer[1]])

            if activ_fcn == 'relu6':
                activ = tf.nn.relu6
            elif activ_fcn == 'elu':
                activ = tf.nn.elu
            elif activ_fcn == 'mixed':
                activ = tf.nn.tanh
            h3 =activ(fc(rnn_output, 'pi_fc1', nh=units_per_hlayer[2]))  # , init_scale=np.sqrt(2)))
            pi_logit = fc(h3, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logit)

            vf = fc(rnn_output, 'vf', 1)[:, 0]  # predicted value of input state

        self.pd = CategoricalPd(pi_logit)  # pdparam
        a0 = self.pd.sample()  # returns action index: 0,1
        # a0 = tf.argmax(pi_logit, axis=1)
        neglogp0 = self.pd.neglogp(a0)

        # The rnn state consists of the "cell state" c and the "input vector" x_t = h_{t-1}
        self.initial_state = (np.zeros([nenvs, units_per_hlayer[1]]), np.zeros([nenvs, units_per_hlayer[1]]))

        def step(ob, r_state, *_args, **_kwargs):
            a, pi, v, r_state_out, neglogp = sess.run([a0, pi_logit, vf, rnn_state_out, neglogp0], {X: ob, rnn_state_in: r_state})
            return a, pi, v, r_state_out, neglogp

        def value(ob, r_state, *_args, **_kwargs):
            return sess.run(vf, {X: ob, rnn_state_in: r_state})

        self.X = X
        self.pi = pi
        self.pi_logit = pi_logit
        self.vf = vf
        self.ac = a0
        self.rnn_state_in = rnn_state_in
        self.rnn_state_out = rnn_state_out
        self.step = step
        self.value = value


class GRUPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenvs, nsteps, units_per_hlayer, reuse=False,
                 activ_fcn='relu6'):  # pylint: disable=W0613
        # this method is called with nbatch = nenvs*nsteps

        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc)
        # actdim = ac_space.shape[0]
        # Todo check initialization
        # Input and Output dimensions
        nd, = ob_space.shape
        nbatch = nenvs * nsteps
        ob_shape = (nbatch, nd)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            if activ_fcn == 'relu6':
                h1 = tf.nn.relu6(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  # , init_scale=np.sqrt(2)))
                h2 = tf.nn.relu6(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  # , init_scale=np.sqrt(2)))
            elif activ_fcn == 'elu':
                h1 = tf.nn.elu(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  # , init_scale=np.sqrt(2)))
                h2 = tf.nn.elu(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  # , init_scale=np.sqrt(2)))
            elif activ_fcn == 'mixed':
                h1 = tf.nn.relu6(fc(X, 'pi_vf_fc1', nh=units_per_hlayer[0]))  # , init_scale=np.sqrt(2)))
                h2 = tf.nn.relu6(fc(h1, 'pi_vf_fc2', nh=units_per_hlayer[1]))  # , init_scale=np.sqrt(2)))

            # The output matrix [nbatch x trace_length, h_units] of layer 2 needs to be reshaped to a vector with
            # dimensions: [nbatch , trace_length , h_units] for rnn processing.
            rnn_cell = tf.contrib.rnn.GRUCell(num_units=units_per_hlayer[1])
            rnn_input = tf.reshape(h2, shape=[nenvs, nsteps, units_per_hlayer[1]])
            rnn_state_in = rnn_cell.zero_state(batch_size=nenvs,
                                               dtype=tf.float32)  # reset the state in every training iteration
            rnn_output, rnn_state_out = tf.nn.dynamic_rnn(inputs=rnn_input,
                                                          cell=rnn_cell,
                                                          initial_state=rnn_state_in,
                                                          dtype=tf.float32,
                                                          scope="model" + '_rnn')
            # The output of the recurrent cell then needs to be reshaped to the original matrix shape.
            rnn_output = tf.reshape(rnn_output, shape=[-1, units_per_hlayer[1]])

            if activ_fcn == 'relu6':
                activ = tf.nn.relu6
            elif activ_fcn == 'elu':
                activ = tf.nn.elu
            elif activ_fcn == 'mixed':
                activ = tf.nn.tanh
            h3 = activ(fc(rnn_output, 'pi_fc1', nh=units_per_hlayer[2]))  # , init_scale=np.sqrt(2)))
            pi_logit = fc(h3, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logit)

            vf = fc(rnn_output, 'vf', 1)[:, 0]  # predicted value of input state

        self.pd = CategoricalPd(pi_logit)  # pdparam
        a0 = self.pd.sample()  # returns action index: 0,1
        # a0 = tf.argmax(pi_logit, axis=1)
        neglogp0 = self.pd.neglogp(a0)

        # The rnn state consists of the "cell state" c and the "input vector" x_t = h_{t-1}
        self.initial_state = np.zeros([nenvs, units_per_hlayer[1]])

        def step(ob, r_state, *_args, **_kwargs):
            a, pi, v, r_state_out, neglogp = sess.run([a0, pi_logit, vf, rnn_state_out, neglogp0], {X: ob, rnn_state_in: r_state})
            return a, pi, v, r_state_out, neglogp

        def value(ob, r_state,  *_args, **_kwargs):
            return sess.run(vf, {X: ob, rnn_state_in: r_state})

        self.X = X
        self.pi = pi
        self.pi_logit = pi_logit
        self.vf = vf
        self.ac = a0
        self.rnn_state_in = rnn_state_in
        self.rnn_state_out = rnn_state_out
        self.step = step
        self.value = value


# -------------------------------------------------------------------------------------------------
#                                           DQN
# -------------------------------------------------------------------------------------------------
class LSTM_DQN():
    """
    Deep Recurrent Q Network class based on TensorFlow.
    """
    def __init__(self, sess, ob_space, nact, nbatch, trace_length, units_per_hlayer, scope=None, reuse=False, activ_fcn='relu6'):
        nd, = ob_space.shape
        nflat_batch = nbatch * trace_length
        ob_shape = [nflat_batch, nd]
        X = tf.placeholder(shape=ob_shape, dtype=tf.float32,
                                     name=scope + "_obs")  # observations
        # Network Architecture
        with tf.variable_scope(scope, reuse=reuse):  # leads to error when assigning weights to target network
            if activ_fcn == 'relu6':
                h1 = tf.nn.relu6(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.relu6(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.relu6(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            elif activ_fcn == 'elu':
                h1 = tf.nn.elu(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.elu(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.elu(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            elif activ_fcn == 'mixed':
                h1 = tf.nn.relu6(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.relu6(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.tanh(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            # The output matrix [nbatch x trace_length, h_units] of layer 3 needs to be reshaped to a vector with
            # dimensions: [nbatch x trace_length x h_units] for rnn processing.
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=units_per_hlayer[2], state_is_tuple=True)
            rnn_input = tf.reshape(h3, shape=[nbatch, trace_length, units_per_hlayer[2]])
            rnn_state_in = rnn_cell.zero_state(batch_size=nbatch, dtype=tf.float32)  # reset the state in every training iteration

            rnn_output, rnn_state_out = tf.nn.dynamic_rnn(inputs=rnn_input,
                                                          cell=rnn_cell,
                                                          initial_state=rnn_state_in,
                                                          dtype=tf.float32,
                                                          scope=scope+'_rnn')
            # The output of the recurrent cell then needs to be reshaped to the original matrix shape.
            rnn_output = tf.reshape(rnn_output, shape=[-1, units_per_hlayer[2]])

            # If dueling Q network: split output of RNN cell into Value and Advantage part
            # Here: only use output as Value function

            # Output: predicted Q values of the best action with linear activation.
            self.predQ = tf.layers.dense(rnn_output, nact, activation=None, kernel_initializer=None)
            a0 = tf.arg_max(self.predQ, dimension=1)

        def predict(obs, rnn_state):
            """
            Args:
                sess: TensorFlow session
                obs: array of observations for which we want to predict the actions. [batch_size]
            Return:
                The prediction of the output tensor. [batch_size, n_valid_actions]
            """
            return sess.run([self.predQ, rnn_state_out], feed_dict={X: obs, rnn_state_in: rnn_state})

        def step(obs_in, rnn_state):
            return sess.run([a0, rnn_state_out], feed_dict={X: obs_in, rnn_state_in: rnn_state})

        def state(obs, rnn_state):
            return sess.run(rnn_state_out, feed_dict={X: obs, rnn_state_in: rnn_state})

        self.initial_state = (np.zeros([nbatch, units_per_hlayer[2]]), np.zeros([nbatch, units_per_hlayer[2]]))
        self.X = X
        self.rnn_state_in = rnn_state_in
        self.rnn_state_out = rnn_state_out
        self.predict = predict
        self.step = step
        self.state = state


class GRU_DQN():
    """
    Deep Recurrent Q Network class based on TensorFlow.
    """
    def __init__(self, sess, ob_space, nact, nbatch, trace_length, units_per_hlayer, scope=None, reuse=False, activ_fcn='relu6'):
        nd, = ob_space.shape
        nflat_batch = nbatch * trace_length
        ob_shape = [nflat_batch, nd]
        X = tf.placeholder(shape=ob_shape, dtype=tf.float32,
                                     name=scope + "_obs")  # observations
        # Network Architecture
        with tf.variable_scope(scope, reuse=reuse):  # leads to error when assigning weights to target network
            if activ_fcn == 'relu6':
                h1 = tf.nn.relu6(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.relu6(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.relu6(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            elif activ_fcn == 'elu':
                h1 = tf.nn.elu(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.elu(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.elu(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            elif activ_fcn == 'mixed':
                h1 = tf.nn.relu6(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.relu6(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.tanh(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            # The output matrix [nbatch x trace_length, h_units] of layer 3 needs to be reshaped to a vector with
            # dimensions: [nbatch x trace_length x h_units] for rnn processing.
            rnn_cell = tf.contrib.rnn.GRUCell(num_units=units_per_hlayer[2])
            rnn_input = tf.reshape(h3, shape=[nbatch, trace_length, units_per_hlayer[2]])
            rnn_state_in = rnn_cell.zero_state(batch_size=nbatch, dtype=tf.float32)  # reset the state in every training iteration

            rnn_output, rnn_state_out = tf.nn.dynamic_rnn(inputs=rnn_input,
                                                          cell=rnn_cell,
                                                          initial_state=rnn_state_in,
                                                          dtype=tf.float32,
                                                          scope=scope+'_rnn')
            # The output of the recurrent cell then needs to be reshaped to the original matrix shape.
            rnn_output = tf.reshape(rnn_output, shape=[-1, units_per_hlayer[2]])

            # If dueling Q network: split output of RNN cell into Value and Advantage part
            # Here: only use output as Value function

            # Output: predicted Q values of the best action with linear activation.
            self.predQ = tf.layers.dense(rnn_output, nact, activation=None, kernel_initializer=None)
            a0 = tf.arg_max(self.predQ, dimension=1)

        def predict(obs, rnn_state):
            """
            Args:
                sess: TensorFlow session
                obs: array of observations for which we want to predict the actions. [batch_size]
            Return:
                The prediction of the output tensor. [batch_size, n_valid_actions]
            """
            return sess.run([self.predQ, rnn_state_out], feed_dict={X: obs, rnn_state_in: rnn_state})

        def step(obs_in, rnn_state):
            return sess.run([a0, rnn_state_out], feed_dict={X: obs_in, rnn_state_in: rnn_state})

        def state(obs, rnn_state):
            return sess.run(rnn_state_out, feed_dict={X: obs, rnn_state_in: rnn_state})

        self.initial_state = np.zeros([nbatch, units_per_hlayer[2]])
        self.X = X
        self.rnn_state_in = rnn_state_in
        self.rnn_state_out = rnn_state_out
        self.predict = predict
        self.step = step
        self.state = state


class FF_DQN():
    """
    Deep Q Network class based on TensorFlow.
    """

    def __init__(self, sess, ob_space, nact, nbatch, trace_length, units_per_hlayer, scope=None, reuse=False, activ_fcn='relu6'):
        nd, = ob_space.shape
        prefix = "target_" if (scope == "target") else ""

        X = tf.placeholder(shape=(nbatch, nd), dtype=tf.float32,
                           name=prefix + "Ob")  # observations
        # Network Architecture
        with tf.variable_scope(scope, reuse=reuse):  # leads to error when assigning weights to target network
            if activ_fcn == 'relu6':
                h1 = tf.nn.relu6(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.relu6(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.relu6(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            elif activ_fcn == 'elu':
                h1 = tf.nn.elu(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.elu(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.elu(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            elif activ_fcn == 'mixed':
                h1 = tf.nn.relu6(fc(X, 'dqn_h1', nh=units_per_hlayer[0]))
                h2 = tf.nn.relu6(fc(h1, 'dqn_h2', nh=units_per_hlayer[1]))
                h3 = tf.nn.tanh(fc(h2, 'dqn_h3', nh=units_per_hlayer[2]))
            predQ = fc(h3, 'predQ', nact, init_scale=0.01)
        a0 = tf.arg_max(predQ, dimension=1)

        def step(obs, *_args, **_kwargs):
            return sess.run(a0, feed_dict={X: obs}), None

        def predict(obs, *_args, **_kwargs):
            """
            Args:
                sess: TensorFlow session
                obs: array of observatons for which we want to predict the actions. [batch_size]
            Return:
                The prediction of the output tensor. [batch_size, n_valid_actions]
            """
            return sess.run(predQ, feed_dict={X: obs}), None

        self.initial_state = None
        self.X = X
        self.predQ = predQ
        self.predict = predict
        self.step = step