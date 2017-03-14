from tf_model import *
from util import *
class Generator(StandardModel):
    def __init__(
            self, config,
            x, x_mask,
            y, y_mask,
            critic):
        self.rewards = tf.placeholder(
                        dtype=tf.float32,
                        name='rewards',
                        shape=(None, None))
        self.critic = critic
        self.reward_is_increment_of_reward = config.reward_is_increment_of_reward 
        self.num_rollouts = config.num_rollouts
        with tf.name_scope("Generator"):
            super(Generator, self).__init__(
                    config,
                    x=x,
                    x_mask=x_mask,
                    y=y,
                    y_mask=y_mask)

            self._build_prefix_score(config)

    def _build_decoder(self, config):
        # use GenDecoder
        with tf.name_scope("encoder"):
            self.ctx = self.encoder.get_context(self.x, self.x_mask)
        with tf.name_scope("decoder"):
            self.decoder = GenDecoder(config, self.ctx, self.x_mask)

    def _build_loss(self, config):
        # Multiply logProbs with rewards
        with tf.name_scope("decoder"):
            self.logits = self.decoder.score(self.y)
        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y,
                    logits=self.logits)
            #cost has shape seqLen x batch
            loss *= self.y_mask
            loss *= self.rewards
            self.loss_per_sentence = tf.reduce_sum(loss, axis=0)
            self.mean_loss = tf.reduce_mean(self.loss_per_sentence, keep_dims=False)

    def _build_optimizer(self, config):
        # Use RMSProp
        generator_vars = tf.get_collection(
                            key=tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope="Generator")
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)
        self.t = tf.Variable(0, name='time', trainable=False, dtype=tf.int32)
        grad_vars = self.optimizer.compute_gradients(
                            self.mean_loss, 
                            var_list=generator_vars)
        grads, varss = zip(*grad_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_c)
        grad_vars = zip(clipped_grads, varss)
        self.apply_grads = self.optimizer.apply_gradients(grad_vars, global_step=self.t)

    def _build_prefix_score(self, config):
        self.prefix_len = tf.placeholder(tf.int32, name='prefix_len', shape=())
        self.rollouts = self.decoder.sample_with_fixed_prefix(self.prefix_len, self.y)
        lengths = tf.reduce_sum(
                    tf.cast(tf.not_equal(self.rollouts, 0), dtype=tf.float32),
                    axis=0)
        lengths = tf.where(tf.equal(self.rollouts[-1], 0), lengths + 1, lengths) #Add 1 for eos if reached
        rollouts_mask = tf.transpose(tf.sequence_mask(lengths, dtype=tf.float32)) 
        scores = self.critic.score(self.rollouts, rollouts_mask, back_prop=False)
        self.prefix_score = scores

    def run_gradient_step_simple(self, sess, x_in, x_mask_in):
        # 1. First generate samples
        # 2. Score every prefix to get reward
        # 3. Use rewards and samples to make the gradient step

        samples, samples_mask = self.generate_fakes(sess, x_in, x_mask_in)
        print 'generated samples', samples.shape
        (seqLen, batch) = samples.shape

        rewards = numpy.zeros(dtype=numpy.float32, shape=(seqLen, batch))
        feeds = {self.x: x_in,
                 self.x_mask: x_mask_in,
                 self.y: samples,
                 self.y_mask: samples_mask,
                 self.prefix_len: None}
        for i in range(1, seqLen + 1):
            feeds[self.prefix_len] = i
            print 'Generating reward for prefix', i,
            for _ in range(self.num_rollouts):
                rewards[i-1] += sess.run(self.prefix_score, feeds)
                print '.',
            print 'Done'
        rewards *= 1./self.num_rollouts 
        if self.reward_is_increment_of_reward and seqLen > 1:
            rewards = np.vstack([rewards[0], rewards[1:] - rewards[:-1]])

        feeds = {self.x: x_in,
                 self.x_mask: x_mask_in,
                 self.y: samples,
                 self.y_mask: samples_mask,
                 self.rewards: rewards}
        sess.run(self.apply_grads, feeds)

    def generate_fakes(self, sess, x_in, x_mask_in):
        samples = sess.run(self._get_samples(), {self.x : x_in, self.x_mask : x_mask_in})
        (seqLen, batch) = samples.shape

        # create mask for samples
        lengths = (samples != 0).sum(axis=0) + 1
        samples_mask = numpy.zeros(dtype=numpy.float32, shape=(seqLen, batch))
        for i in range(batch):
            samples_mask[:lengths[i], i] = 1

        return samples, samples_mask


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

class Critic(StandardModel):
    def __init__(
            self, config,
            x, x_mask, 
            y, y_mask):

        self.labels = tf.placeholder(
                        dtype=tf.float32,
                        name='labels',
                        shape=(None,))
        with tf.name_scope("Critic"):
            super(Critic, self).__init__(
                    config,
                    x=x,
                    x_mask=x_mask,
                    y=y,
                    y_mask=y_mask)

    def _build_decoder(self, config):
        with tf.name_scope("encoder"):
            self.ctx = self.encoder.get_context(self.x, self.x_mask)
        with tf.name_scope("decoder"):
            self.decoder = CriticDecoder(config, self.ctx, self.x_mask)

    def _build_loss(self, config):
        with tf.name_scope("decoder"):
            scores = self.decoder.score(
                            self.y,
                            self.y_mask,
                            back_prop=True)
            loss = scores * self.labels # labels are -1, 1
            # assume same number of true and fake
            # reduce_mean divides sum by (n_true+n_fake) == 2*n_true == 2*n_fake
            self.mean_loss = -2*tf.reduce_mean(loss)

    def _build_optimizer(self, config):
        # Use RMSProp
        critic_vars = tf.get_collection(
                            key=tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope="Critic")
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)
        self.t = tf.Variable(0, name='time', trainable=False, dtype=tf.int32)
        grad_vars = self.optimizer.compute_gradients(
                            self.mean_loss, 
                            var_list=critic_vars)
        grads, varss = zip(*grad_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_c)
        grad_vars = zip(clipped_grads, varss)
        self.apply_grads = self.optimizer.apply_gradients(grad_vars, global_step=self.t)
        with tf.control_dependencies([self.apply_grads]):
            self.clip_vars = [v.assign(tf.clip_by_value(v, -config.weight_clip, config.weight_clip)) \
                                for v in critic_vars]

    def run_gradient_step_simple(
            self, sess,
            x_in, x_mask_in,
            true_in, true_mask_in,
            fake_in, fake_mask_in):
        both_in, both_mask_in, labels_in = merge_batches(
                                            true_in, true_mask_in,
                                            fake_in, fake_mask_in)
        x_in, x_mask_in, _ = merge_batches(
                                x_in, x_mask_in,
                                x_in, x_mask_in)
        inn = {self.x: x_in,
               self.x_mask: x_mask_in,
               self.y: both_in,
               self.y_mask: both_mask_in,
               self.labels: labels_in}
        _, mean_loss, _ = sess.run([self.apply_grads, self.mean_loss, self.clip_vars], inn)
        return mean_loss

    def score(self, y, y_mask, back_prop):
        return self.decoder.score(y, y_mask, back_prop)

class CriticDecoder(Decoder):    
    def __init__(self, config, context, x_mask):
        super(CriticDecoder, self).__init__(config, context, x_mask)

        with tf.name_scope("state_to_score"):
            self.state_to_hidden_layer = FeedForwardLayer(
                                            in_size=config.state_size,
                                            out_size=config.state_size)
            self.hidden_to_score_layer = FeedForwardLayer(
                                            in_size=config.state_size,
                                            out_size=1,
                                            non_linearity=lambda x: x)

    def score(self, y, y_mask, back_prop):
        with tf.name_scope("y_embeddings_layer"):
            seq_len = tf.shape(y)[0]
            y_but_last = tf.slice(y, [0,0], [seq_len - 1, -1])
            y_embs = self.y_emb_layer.forward(y_but_last)
            y_embs = tf.pad(y_embs,
                            mode='CONSTANT',
                            paddings=[[1,0],[0,0],[0,0]]) # prepend zeros

        gates_x, proposal_x = self.grustep1.precompute_from_x(y_embs)
        # Replace with tf.while_loop

        def cond(i, prev_state):
            return tf.less(i, seq_len)

        def body(i, prev_state):
            gates_x2d = gates_x[i]
            proposal_x2d = proposal_x[i]
            state = self.grustep1.forward(
                        prev_state,
                        gates_x=gates_x2d,
                        proposal_x=proposal_x2d)
            att_ctx = self.attstep.forward(state) 
            state = self.grustep2.forward(state, att_ctx)
            # Use y_mask here - because the last state must be correct
            state = tf.where(tf.equal(y_mask[i], 0.), prev_state, state)
            return i+1, state

        #TODO: init_state is only for batch, not 2*batch
        _, last_state = tf.while_loop(
                            cond=cond,
                            body=body,
                            loop_vars=[tf.constant(0), self.init_state],
                            back_prop=back_prop)

        hidden = self.state_to_hidden_layer.forward(last_state)
        score = self.hidden_to_score_layer.forward(hidden)
        score = tf.squeeze(score, axis=1)
        return score

