from tf_model import *
from util import *
import logging
class Generator(StandardModel):
    def __init__(
            self, config,
            x, x_mask,
            y, y_mask):

        self.reward_is_increment_of_reward = config.reward_is_increment_of_reward 
        self.num_rollouts = config.num_rollouts
        with tf.name_scope("Generator"):
            super(Generator, self).__init__(
                    config,
                    x=x,
                    x_mask=x_mask,
                    y=y,
                    y_mask=y_mask)
        self._samples = self._get_samples()
        self._samples_mask = create_samples_mask(self.get_samples())
        self.rewards = None

    def _build_decoder(self, config):
        # use GenDecoder
        with tf.name_scope("encoder"):
            self.ctx = self.encoder.get_context(self.x, self.x_mask)
        with tf.name_scope("decoder"):
            self.decoder = GenDecoder(config, self.ctx, self.x_mask)

    def _build_loss(self, config):
        pass
    def _build_optimizer(self, config):
        print 'This is the fake optimizer'
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)
        self.time = tf.Variable(0, name='time', trainable=False, dtype=tf.int32)
    def add_rewards(self, config, rewards):
        self.rewards = rewards
        with tf.name_scope("Generator"):
            self._build_loss_with_rewards(config, rewards)
            self._build_optimizer_with_rewards(config, rewards)
    def _build_loss_with_rewards(self, config, rewards):
        # Multiply logProbs with rewards
        with tf.name_scope("decoder"):
            self.logits = self.decoder.score(self.get_samples())
        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.get_samples(),
                    logits=self.logits)
            #cost has shape seqLen x batch
            loss *= self.get_samples_mask()
            self.loss_per_sentence = tf.reduce_sum(loss, axis=0)
            if config.use_rollouts:
                loss *= rewards 
                self.loss_per_sentence_with_rewards = tf.reduce_sum(loss, axis=0)
            else:
                self.loss_per_sentence_with_rewards = tf.reduce_sum(loss, axis=0)
                self.loss_per_sentence_with_rewards *= rewards

            self.mean_loss = tf.reduce_mean(self.loss_per_sentence_with_rewards, keep_dims=False)

    def _build_optimizer_with_rewards(self, config, rewards):
        # Use RMSProp?
        generator_vars = tf.get_collection(
                            key=tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope="Generator")
        grad_vars = self.optimizer.compute_gradients(
                            self.mean_loss, 
                            var_list=generator_vars)
        grads, varss = zip(*grad_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_c)
        grad_vars = zip(clipped_grads, varss)
        self.apply_grads = self.optimizer.apply_gradients(grad_vars, global_step=self.time)

    #def _build_prefix_score(self, config, critic):
    #    self.prefix_len = tf.placeholder(tf.int32, name='prefix_len', shape=())
    #    self.rollouts = self.decoder.sample_with_fixed_prefix(self.prefix_len, self.y)
    #    rollouts_mask = create_samples_mask(self.rollouts)
    #    # TODO: Critic does not have a score function atm
    #    scores = critic.score(self.rollouts, rollouts_mask, back_prop=False)
    #    self.prefix_score = scores

    def run_gradient_step_simple(self, sess, config, x_in, x_mask_in):

        if config.use_rollouts:
            # 1. First generate samples
            # 2. Score every prefix to get reward
            # 3. Use rewards and samples to make the gradient step
            assert False, 'Not implemented yet'
            #rewards = numpy.zeros(dtype=numpy.float32, shape=(seqLen, batch))
            #feeds = {self.x: x_in,
            #        self.x_mask: x_mask_in,
            #        self.y: samples,
            #        self.y_mask: samples_mask,
            #        self.prefix_len: None}
            #for i in range(1, seqLen + 1):
            #    feeds[self.prefix_len] = i
            #    print 'Generating reward for prefix', i,
            #    for _ in range(self.num_rollouts):
            #        rewards[i-1] += sess.run(self.prefix_score, feeds)
            #        print '.',
            #    print 'Done'
            #rewards *= 1./self.num_rollouts 
            #if self.reward_is_increment_of_reward and seqLen > 1:
            #    rewards = np.vstack([rewards[0], rewards[1:] - rewards[:-1]])

            #feeds = {self.x: x_in,
            #        self.x_mask: x_mask_in,
            #        self.y: samples,
            #        self.y_mask: samples_mask,
            #        self.rewards: rewards}
            #sess.run(self.apply_grads, feeds)
        else:
            feeds = {self.x: x_in,
                    self.x_mask: x_mask_in}
            _, lps, lps_no_rewards, rewards, samples, samples_mask = sess.run([self.apply_grads,
                                                    self.loss_per_sentence_with_rewards,
                                                    self.loss_per_sentence, 
                                                    self.rewards, self.get_samples(), self.get_samples_mask()], feeds)
            #### DEBUG ####
            source = list(x_in.T)
            samples = list(samples.T)
            samples_mask = list(samples_mask.T)
            source_to_num, target_to_num, num_to_source, num_to_target = load_dictionaries(config)
            assert len(source) == len(samples) == len(samples_mask), "source: {} samples: {} mask: {}".format(len(source), len(samples), len(samples_mask))
            for i, (src, sample, sample_mask, reward) in enumerate(zip(source, samples, samples_mask, rewards)):
                logging.debug("{} SRC: {}".format(i, seqs2words(src, num_to_source)))
                logging.debug("{} GEN: {} Length: {}".format(i, seqs2words(sample, num_to_target), len(sample)))
                logging.debug("{} RWD: {}".format(i, reward))
                logging.debug("{} MSK: {} Length: {}".format(i, sample_mask, len(sample_mask)))
                logging.debug("{} GENnum: {}".format(i, sample))
                logging.debug("{} MSKnum: {}".format(i, sample_mask))

            return lps, rewards, lps_no_rewards
    def get_samples(self):
        return self._samples
    def get_samples_mask(self):
        return self._samples_mask

class GenDecoder(Decoder):
    def __init__(self, config, context, x_mask):
        super(GenDecoder, self).__init__(config, context, x_mask)

    def sample_with_fixed_prefix(self, prefix_len, samples):
        batch_size = tf.shape(self.init_state)[0]
        i = tf.constant(0)
        init_ys = -tf.ones(dtype=tf.int32, shape=[batch_size])
        init_embs = tf.zeros(dtype=tf.float32, shape=[batch_size,self.embedding_size])
        ys_array = tf.TensorArray(
                     dtype=tf.int32,
                     size=self.translation_maxlen,
                     clear_after_read=True, #TODO: does this help? or will it only introduce bugs in the future?
                     name='y_sampled_array')
        init_loop_vars = [i, self.init_state, init_ys, init_embs, ys_array]
        def cond(i, states, prev_ys, prev_embs, ys_array):
            return tf.logical_and(
                    tf.less(i, self.translation_maxlen),
                    tf.reduce_any(tf.not_equal(prev_ys, 0)))

        def body(i, states, prev_ys, prev_embs, ys_array):
            new_states1 = self.grustep1.forward(states, prev_embs)
            att_ctx = self.attstep.forward(new_states1)
            new_states2 = self.grustep2.forward(new_states1, att_ctx)

            def sampleNew():
                logits = self.predictor.get_logits(prev_embs, new_states2, att_ctx, multi_step=False)
                new_ys = tf.multinomial(logits, num_samples=1)
                new_ys = tf.cast(new_ys, dtype=tf.int32)
                new_ys = tf.squeeze(new_ys, axis=1)
                return new_ys

            def copyOld():
                new_ys = samples[i]
                return new_ys
            new_ys = tf.cond(i < prefix_len, copyOld, sampleNew)
            new_ys = tf.where(
                    tf.equal(prev_ys, tf.constant(0, dtype=tf.int32)),
                    tf.zeros_like(new_ys),
                    new_ys)
            ys_array = ys_array.write(index=i, value=new_ys)
            new_embs = self.y_emb_layer.forward(new_ys)
            return i+1, new_states2, new_ys, new_embs, ys_array

        final_loop_vars = tf.while_loop(
                            cond=cond,
                            body=body,
                            loop_vars=init_loop_vars,
                            back_prop=False)
        i, _, _, _, ys_array = final_loop_vars
        sampled_ys = ys_array.gather(tf.range(0, i))

        return sampled_ys


