import sabre
import math
import numpy as np
import a3c
import tensorflow as tf
#import tflearn

RAND_RANGE = 1000


class Zero(sabre.Abr):
    S_INFO = 6
    S_LEN = 20
    THROUGHPUT_NORM = 5 * 1024.0
    TIME_NORM = 10.0
   # A_DIM = len(VIDEO_BIT_RATE)
    ACTOR_LR_RATE = 1e-4
    CRITIC_LR_RATE = 1e-3
    A_DIM = 10
    GRADIENT_BATCH_SIZE = 32

    def __init__(self, scope):
        # self.gp = config['gp']
        # self.buffer_size = config['buffer_size']
        # self.abr_osc = config['abr_osc']
        # self.abr_basic = config['abr_basic']
        self.quality = 0
        #self.last_quality = 0
        self.state = np.zeros((Zero.S_INFO, Zero.S_LEN))
        self.quality_len = Zero.A_DIM
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        self.dual = a3c.DualNetwork(self.sess, scope)
        self.actor = a3c.ActorNetwork(self.sess,
                                      state_dim=[
                                          Zero.S_INFO, Zero.S_LEN], action_dim=self.quality_len,
                                      learning_rate=Zero.ACTOR_LR_RATE, scope=scope, dual=self.dual)
        self.critic = a3c.CriticNetwork(self.sess,
                                        state_dim=[Zero.S_INFO, Zero.S_LEN],
                                        learning_rate=Zero.CRITIC_LR_RATE, scope=scope, dual=self.dual)
        self.sess.run(tf.global_variables_initializer())
        self.history = []
        self.replay_buffer = []
        # self.s_batch = [np.zeros((Zero.S_INFO, Zero.S_LEN))]
        # action_vec = np.zeros(Zero.A_DIM)
        # self.a_batch = [action_vec]
        # self.r_batch = []
        # self.actor_gradient_batch = []
        # self.critic_gradient_batch = []

    def clear(self):
        self.replay_buffer = []

    def learn(self, buffer = None):
        actor_gradient_batch, critic_gradient_batch = [], []
        if buffer is None:
            _buf = self.replay_buffer
        else:
            _buf = buffer
        for (s_batch, a_batch, r_batch) in _buf:
            actor_gradient, critic_gradient, td_batch = \
                a3c.compute_gradients(s_batch=np.stack(s_batch),
                                      a_batch=np.vstack(a_batch),
                                      r_batch=np.vstack(r_batch),
                                      actor=self.actor, critic=self.critic)

            actor_gradient_batch.append(actor_gradient)
            critic_gradient_batch.append(critic_gradient)

        for i in range(len(actor_gradient_batch)):
            self.actor.apply_gradients(actor_gradient_batch[i])
            self.critic.apply_gradients(critic_gradient_batch[i])

        self.actor_gradient_batch = []
        self.critic_gradient_batch = []
        self.replay_buffer = []

    def pull(self):
        return self.replay_buffer

    def push(self, reward):
        s_batch, a_batch, r_batch = [], [], []
        for (state, action) in self.history:
            s_batch.append(state)
            action_vec = np.zeros(Zero.A_DIM)
            action_vec[action] = 1
            a_batch.append(action_vec)
            r_batch.append(reward)
        self.replay_buffer.append((s_batch, a_batch, r_batch))
        # actor_gradient, critic_gradient, td_batch = \
        #     a3c.compute_gradients(s_batch=np.stack(self.s_batch),
        #                           a_batch=np.vstack(self.a_batch),
        #                           r_batch=np.vstack(self.r_batch),
        #                           actor=self.actor, critic=self.critic)
        # td_loss = np.mean(td_batch)

        # self.actor_gradient_batch.append(actor_gradient)
        # self.critic_gradient_batch.append(critic_gradient)

        self.history = []

    def get_quality_delay(self, segment_index):
        #print(self.buffer_size, sabre.manifest.segment_time, sabre.get_buffer_level(),sabre.manifest.segments[segment_index])
        # print(sabre.log_history[-1],sabre.throughput)
        manifest_len = len(sabre.manifest.segments)
        time, throughput, latency, quality, rebuffer_time = sabre.log_history[-1]
        state = self.state
        state = np.roll(state, -1, axis=1)
        state[0, -1] = throughput / Zero.THROUGHPUT_NORM
        state[1, -1] = time / Zero.TIME_NORM
        state[2, -1] = latency / 1000.0
        state[3, -1] = quality / self.quality_len
        state[4, -1] = rebuffer_time / Zero.TIME_NORM
        state[5, -1] = manifest_len - segment_index

        self.state = state
        action_prob = self.actor.predict(
            np.reshape(state, (1, Zero.S_INFO, Zero.S_LEN)))
        # print(action_prob[0])
        #quality = np.argmax(action_prob[0])
        action_cumsum = np.cumsum(action_prob[0])
        quality = (action_cumsum > np.random.randint(
            1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        delay = 0
        self.history.append((self.state, quality))
        return (quality, delay)

    def update_gradients(self, reward):
        pass
