import sabre
import math
import numpy as np
import dual as a3c
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#import tflearn

RAND_RANGE = 10000
EPS = 1e-6


class Zero(sabre.Abr):
    S_INFO = 9
    S_LEN = 10
    THROUGHPUT_NORM = 5 * 1024.0
    BITRATE_NORM = 8 * 1024.0
    TIME_NORM = 1000.0
   # A_DIM = len(VIDEO_BIT_RATE)
    ACTOR_LR_RATE = 1e-4
    CRITIC_LR_RATE = 1e-3
    GAN_LR_RATE = 1e-4
    A_DIM = 10
    D_DIM = 5
    D_STEP = 0.5
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
        self.delay_len = Zero.D_DIM
        self.delay = np.zeros((self.delay_len))
        for p in range(Zero.D_DIM):
            self.delay[p] = Zero.D_STEP * p

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        # self.gan = a3c.GANNetwork(self.sess, state_dim=[
        #    Zero.S_INFO, Zero.S_LEN], learning_rate=Zero.GAN_LR_RATE, scope=scope)
        self.dual = a3c.DualNetwork(self.sess, scope)
        self.actor = a3c.ActorNetwork(self.sess,
                                      state_dim=[
                                          Zero.S_INFO, Zero.S_LEN], action_dim=self.quality_len * self.delay_len,
                                      learning_rate=Zero.ACTOR_LR_RATE, scope=scope,
                                      dual=self.dual)  # , gan=self.gan)
        self.critic = a3c.CriticNetwork(self.sess,
                                        state_dim=[Zero.S_INFO, Zero.S_LEN],
                                        learning_rate=Zero.CRITIC_LR_RATE, scope=scope,
                                        dual=self.dual)  # , gan=self.gan)
        self.sess.run(tf.global_variables_initializer())
        self.history = []
        self.quality_history = []
        self.replay_buffer = []
        # self.s_batch = [np.zeros((Zero.S_INFO, Zero.S_LEN))]
        # action_vec = np.zeros(Zero.A_DIM)
        # self.a_batch = [action_vec]
        # self.r_batch = []
        # self.actor_gradient_batch = []
        # self.critic_gradient_batch = []

    def teach(self, buffer):
        for (s_batch, a_batch, r_batch) in buffer:
            _s = np.array(s_batch)
            _a = np.array(a_batch)
            #print(_s.shape, _a.shape)
            # for (_s, _a, _r) in zip(s_batch, a_batch, r_batch):
            #     _s = np.reshape(_s, (1, Zero.S_INFO, Zero.S_LEN))
            #     _a = np.reshape(_a, (1, Zero.A_DIM))
            #     if _r > 0:
            self.actor.teach(_s, _a)
            #self.replay_buffer.append((s, a, r))

    def clear(self):
        self.history = []
        self.replay_buffer = []

    def learn(self, ratio=1.0):
        #print(len(self.replay_buffer))
        actor_gradient_batch, critic_gradient_batch = [], []

        for (s_batch, a_batch, r_batch) in self.replay_buffer:
            actor_gradient, critic_gradient, td_batch = \
                a3c.compute_gradients(s_batch=np.stack(s_batch, axis=0),
                                      a_batch=np.vstack(a_batch),
                                      r_batch=np.vstack(r_batch),
                                      actor=self.actor, critic=self.critic,
                                      lr_ratio=ratio)
            #self.gan.optimize(s_batch, r_batch)

            actor_gradient_batch.append(actor_gradient)
            critic_gradient_batch.append(critic_gradient)
            
        for i in range(len(actor_gradient_batch)):
            self.actor.apply_gradients(actor_gradient_batch[i], lr_ratio=ratio)
            self.critic.apply_gradients(
                critic_gradient_batch[i], lr_ratio=ratio)

        self.actor_gradient_batch = []
        self.critic_gradient_batch = []

    def pull(self):
        _ret_buffer = []
        for (s_batch, a_batch, r_batch) in self.replay_buffer:
            if r_batch[-1] > 0:
                _ret_buffer.append((s_batch, a_batch, r_batch))
        return _ret_buffer

    def push(self, reward):
        s_batch, a_batch, r_batch = [], [], []
        _index = 0
        for (state, action) in self.history:
            s_batch.append(state)
            action_vec = np.zeros(Zero.A_DIM * Zero.D_DIM)
            action_vec[action] = 1
            a_batch.append(action_vec)
            r_batch.append(reward[_index])
            _index += 1

        self.replay_buffer.append((s_batch, a_batch, r_batch))

        self.history = []
        self.quality_history = []
        self.state = np.zeros((Zero.S_INFO, Zero.S_LEN))

    def _get_quality_delay(self, action):
        return action // Zero.A_DIM, action % Zero.A_DIM

    def get_quality_delay(self, segment_index):
        #print(self.buffer_size, sabre.manifest.segment_time, sabre.get_buffer_level(),sabre.manifest.segments[segment_index])
        # print(sabre.log_history[-1],sabre.throughput)
        if segment_index != 0:
            self.quality_history.append(
                (sabre.played_bitrate, sabre.rebuffer_time, sabre.total_bitrate_change))
        if segment_index < 0:
            return
        manifest_len = len(sabre.manifest.segments) * sabre.manifest.segment_time
        time, throughput, latency, quality, _ = sabre.log_history[-1]
        state = self.state
        state = np.roll(state, -1, axis=1)
        state[0, -1] = throughput / Zero.THROUGHPUT_NORM
        state[1, -1] = time / manifest_len
        state[2, -1] = sabre.total_play_time / manifest_len
        state[3, -1] = latency / Zero.TIME_NORM
        state[4, -1] = quality / self.quality_len
        #state[5, -1] = sabre.played_bitrate / \
        #    (segment_index * np.max(sabre.manifest.bitrates))
        #state[6, -1] = sabre.rebuffer_time / (sabre.total_play_time + EPS)
        #state[7, -1] = sabre.total_bitrate_change / \
        #    (sabre.played_bitrate + EPS)
        #state[6, -1] = (manifest_len - segment_index) / manifest_len
        #state[7, -1] = sabre.get_buffer_level() / 25.0
        state[5, 0:Zero.A_DIM] = np.array(sabre.manifest.bitrates /
                                          np.max(sabre.manifest.bitrates))
        state[6, 0:Zero.A_DIM] = np.array(
            sabre.manifest.segments[segment_index]) / 1024.0 / 1024.0 / 2.0
        _fft = np.fft.fft(state[0])
        state[7] = _fft.real
        state[8] = _fft.imag

        self.state = state
        action_prob = self.actor.predict(
            np.reshape(state, (1, Zero.S_INFO, Zero.S_LEN)))
        # print(action_prob[0])
        #quality = np.argmax(action_prob[0])
        action_cumsum = np.cumsum(action_prob[0])
        action = (action_cumsum > np.random.randint(
            1, RAND_RANGE) / float(RAND_RANGE)).argmax()

        quality, delay = self._get_quality_delay(action)
        _delay = delay * Zero.D_STEP
        self.history.append((self.state, action))
        return (quality, _delay)
