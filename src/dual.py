import numpy as np
import tensorflow as tf
import tflearn


GAMMA = 0.99
ENTROPY_WEIGHT = 0.1
ENTROPY_EPS = 1e-6
FEATURE_NUM = 64
KERNEL = 3


class DualNetwork(object):
    def __init__(self, sess, scope):
        self.sess = sess
        self.scope = scope
        self.reuse = False

    def attention(self, inputs, attention_size):
        # the length of sequences processed in the antecedent RNN layer
        inputs = tf.stack(inputs, axis=1)
        sequence_length = inputs.get_shape()[1].value
        # hidden size of the RNN layer
        hidden_size = inputs.get_shape()[2].value

        # Attention mechanism
        W_omega = tf.Variable(tf.random_normal(
            [hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(
            inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(
            inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        return output, alphas

    def CNN_Core(self, x, reuse=False):
        with tf.variable_scope(self.scope + '-cnn_core', reuse=reuse):
            tmp = tflearn.conv_1d(
                x, FEATURE_NUM // 4, KERNEL, activation='relu')
            tmp = tflearn.batch_normalization(tmp)
            tmp = tflearn.flatten(tmp)
            return tmp

    def create_dual_network(self, inputs, s_dim):
        with tf.variable_scope(self.scope + '-dual', reuse=self.reuse):
            split_array = []
            for i in range(s_dim[0]):
                tmp = tf.reshape(inputs[:, i:i+1, :], (-1, s_dim[1], 1))
                branch1 = tflearn.conv_1d(
                    tmp, FEATURE_NUM, 3, activation='relu')
                branch2 = tflearn.conv_1d(
                    tmp, FEATURE_NUM, 4, activation='relu')
                branch3 = tflearn.conv_1d(
                    tmp, FEATURE_NUM, 5, activation='relu')
                network = tflearn.merge([branch1, branch2, branch3], mode='concat', axis=1)
                network = tf.expand_dims(network, 2)
                network = tflearn.global_avg_pool(network)
                split_array.append(network)
            #out, _ = self.attention(split_array, FEATURE_NUM)
            out = tflearn.merge(split_array, 'concat')
            self.reuse = True
            return out


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, scope, dual):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        #self.lr_rate = learning_rate
        self.learning_rate = learning_rate
        self.basic_entropy = ENTROPY_WEIGHT
        # self.sess.run(self.lr_rate, feed_dict={
        #              self.lr_rate: self.basic_learning_rate})
        self.scope = scope
        self.dual = dual

        # Create the actor network
        self.inputs, self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope=self.scope + '-actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.y_ = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])
        self.lr_rate = tf.placeholder(tf.float32)
        self.entropy = tf.placeholder(tf.float32)

        self.loss = tflearn.objectives.softmax_categorical_crossentropy(
            self.out, self.y_)
        self.teach_op = tf.train.AdamOptimizer(
            learning_rate=self.lr_rate).minimize(self.loss)

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(
            tf.multiply(
                tf.log(
                    tf.reduce_sum(
                        tf.multiply(self.out, self.acts),
                        reduction_indices=1,
                        keep_dims=True
                    )
                ),
                -self.act_grad_weights
            )
        ) \
            + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out,
                                                       tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope(self.scope + '-actor'):
            inputs = tflearn.input_data(
                shape=[None, self.s_dim[0], self.s_dim[1]])
        dense_net_0 = self.dual.create_dual_network(inputs, self.s_dim)
        with tf.variable_scope(self.scope + '-actor'):
            dense_net_0 = tflearn.fully_connected(
                dense_net_0, FEATURE_NUM, activation='relu')
            out = tflearn.fully_connected(
                dense_net_0, self.a_dim, activation='softmax')

            return inputs, out

    def train(self, inputs, acts, act_grad_weights):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights, lr_ratio=1.0):
        #_entropy = self.basic_entropy * \
        #    (lr_ratio - 1.0 + ENTROPY_EPS) * np.log(lr_ratio + ENTROPY_EPS)
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
            #self.entropy: _entropy
            #self.lr_rate: _lr
        })

    def apply_gradients(self, actor_gradients, lr_ratio = 1.0):
        _dict = {}
        for i, d in zip(self.actor_gradients, actor_gradients):
            _dict[i] = d
        _lr = self.learning_rate * \
            (lr_ratio - 1.0 + ENTROPY_EPS) * np.log(lr_ratio + ENTROPY_EPS)
        _dict[self.lr_rate] = _lr
        return self.sess.run(self.optimize, feed_dict=_dict)

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def teach(self, state, action):
        return self.sess.run(self.teach_op, feed_dict={
            self.inputs: state, self.y_: action
        })


class RudderNetwork(object):
    #SEQ_LEN = 20
    def __init__(self, sess, state_dim, learning_rate, scope):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.scope = scope
        self.s_dim_queue = []

    def create_rudder_network(self):
        pass


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, learning_rate, scope, dual):
        self.sess = sess
        self.s_dim = state_dim
        #self.lr_rate = learning_rate
        self.learning_rate = learning_rate
        self.scope = scope
        self.dual = dual

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope=self.scope + '-critic')

        self.lr_rate = tf.placeholder(tf.float32)
        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope(self.scope + '-critic'):
            inputs = tflearn.input_data(
                shape=[None, self.s_dim[0], self.s_dim[1]])
        dense_net_0 = self.dual.create_dual_network(inputs, self.s_dim)
        with tf.variable_scope(self.scope + '-critic'):
            dense_net_0 = tflearn.fully_connected(
                dense_net_0, FEATURE_NUM, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')
            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients, lr_ratio = 1.0):
        _dict = {}
        for i, d in zip(self.critic_gradients, critic_gradients):
            _dict[i] = d
        
        _lr = self.learning_rate * \
            (lr_ratio - 1.0 + ENTROPY_EPS) * np.log(lr_ratio + ENTROPY_EPS)
        _dict[self.lr_rate] = _lr
        return self.sess.run(self.optimize, feed_dict=_dict)

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    # def teach(self, state, action):
    #     return self.sess.run(self.teach, feed_dict={
    #         self.inputs: state, self.y_: action
    #     })


class GANNetwork(object):
    # https://arxiv.org/pdf/1406.2661.pdf
    #disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
    #gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    def __init__(self, sess, state_dim, learning_rate, scope, dual, critic):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.scope = scope
        self.dual = dual
        self.critic = critic
        self.inputs_g, self.generate = self.create_generate_network()
        self.inputs_d, self.discriminator = self.create_discriminator_network()
        self.out = tf.placeholder(tf.float32, [None, 1])

        self.gen_loss = -tf.reduce_mean(tf.log(self.discriminator))
        self.disc_loss = tflearn.mean_square(
            tf.log(self.discriminator), self.out)

    def create_generate_network(self):
        with tf.variable_scope(self.scope + '-gan_g'):
            inputs = tflearn.input_data(
                shape=[None, self.s_dim[0], self.s_dim[1]])
        dense_net_0 = self.dual.create_dual_network(inputs, self.s_dim)
        with tf.variable_scope(self.scope + '-gan_g-end'):
            out = tflearn.fully_connected(
                dense_net_0, FEATURE_NUM, activation='sigmoid')
            return inputs, out

    def create_discriminator_network(self):
        with tf.variable_scope(self.scope + '-gan_d'):
            inputs = tflearn.input_data(shape=[None, FEATURE_NUM])
            net = tflearn.fully_connected(inputs, 128, activation='relu')
            net = tflearn.fully_connected(inputs, 64, activation='relu')
            out = tflearn.fully_connected(net, 1, activation='linear')
            return inputs, out


def compute_gradients(s_batch, a_batch, r_batch, actor, critic, lr_ratio=1.0):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)
    R_batch = np.zeros(r_batch.shape)

    # if terminal:
    #    R_batch[-1, 0] = 0  # terminal state
    # else:
    R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch, lr_ratio)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
