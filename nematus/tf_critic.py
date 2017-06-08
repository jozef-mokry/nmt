from tf_model import *
from util import *
import logging
class Critic(StandardModel):
    def __init__(
            self, config,
            x, x_mask, 
            y, y_mask,
            samples, samples_mask):

        self.samples = samples
        self.samples_mask = samples_mask
        self.config = config
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
            self.true_scores, self.fake_scores = self.decoder.score_true_and_fake(
                            config,
                            self.y,
                            self.y_mask,
                            self.samples,
                            self.samples_mask)
            if config.sigmoid_score:
                # log loss and sigmoid
                self.true_loss = tf.log(self.true_scores)
                self.fake_loss = -tf.log(1. - self.fake_scores)
            else:
                # wasserstein loss
                self.true_loss = self.true_scores
                self.fake_loss = self.fake_scores
            self.mean_true_loss = tf.reduce_mean(self.true_loss)
            self.mean_fake_loss = tf.reduce_mean(self.fake_loss)
            self.mean_loss = self.mean_fake_loss - self.mean_true_loss

    def _build_optimizer(self, config):
        critic_vars = tf.get_collection(
                            key=tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope="Critic")
        if config.use_adam:
            logging.info('Optimizer for critic is Adam')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        else:
            logging.info('Optimizer for critic is RMSprop')
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)
        self.time = tf.Variable(0, name='time', trainable=False, dtype=tf.int32)
        grad_vars = self.optimizer.compute_gradients(
                            self.mean_loss, 
                            var_list=critic_vars)
        grads, varss = zip(*grad_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_c)
        grad_vars = zip(clipped_grads, varss)
        self.apply_grads = self.optimizer.apply_gradients(grad_vars, global_step=self.time)
        if config.weight_clip > 0:
            logging.info("Weights will be clipped to {}".format(config.weight_clip))
            with tf.control_dependencies([self.apply_grads]):
                self.clip_vars = [v.assign(tf.clip_by_value(v, -config.weight_clip, config.weight_clip)) \
                                    for v in critic_vars]
        else:
            logging.info("Weights will not be clipped")
            self.clip_vars = []

    def run_gradient_step(
            self, sess,
            x_in, x_mask_in,
            true_in, true_mask_in):
        inn = {self.x: x_in,
               self.x_mask: x_mask_in,
               self.y: true_in,
               self.y_mask: true_mask_in}
        _, _, mean_loss, mean_true_loss, mean_fake_loss = sess.run(
                [self.apply_grads,
                 self.clip_vars,
                 self.mean_loss,
                 self.mean_true_loss,
                 self.mean_fake_loss], inn)
        return mean_loss, mean_true_loss, mean_fake_loss

    def get_true_scores(self):
        return self.true_scores
    def get_fake_scores(self):
        return self.decoder.score(self.config, self.get_samples(), self.get_samples_mask())
    def get_x(self):
        return self.x
    def get_x_mask(self):
        return self.x_mask
    def get_y(self):
        return self.y
    def get_y_mask(self):
        return self.y_mask
    def get_samples(self):
        return self.samples
    def get_samples_mask(self):
        return self.samples_mask

class CriticDecoder(Decoder):    
    def __init__(self, config, context, x_mask):
        super(CriticDecoder, self).__init__(config, context, x_mask)

        with tf.name_scope("state_to_score"):
            self.hidden_layers = []
            for i in range(1, config.num_layers):
                with tf.name_scope("hidden_" + str(i)):
                    logging.debug("Creating hidden layer {}".format(i))
                    self.hidden_layers.append(FeedForwardLayer(
                                                in_size=config.state_size,
                                                out_size=config.state_size,
                                                non_linearity=tf.nn.tanh))

            if config.sigmoid_score:
                non_linearity = tf.nn.sigmoid
            else:
                non_linearity = lambda x: x
            self.last_hidden_to_score_layer = FeedForwardLayer(
                                            in_size=config.state_size,
                                            out_size=1,
                                            non_linearity=non_linearity)

    def _pad_and_concat(self, x, y):
        #concat x and y along axis=1, and pad to same length along axis0
        seq_len_x = tf.shape(x)[0]
        seq_len_y = tf.shape(y)[0]

        def pad_x():
            x_pad = tf.pad(x, [[0,seq_len_y - seq_len_x], [0,0]], mode='CONSTANT')
            return x_pad, y
        def pad_y():
            y_pad = tf.pad(y, [[0,seq_len_x - seq_len_y], [0,0]], mode='CONSTANT')
            return x, y_pad
        x, y = tf.cond(seq_len_x < seq_len_y, pad_x, pad_y)
        out = tf.concat([x,y], axis=1)
        return out

    def _repeat_init_state(self):
        self.init_state_rep = tf.concat([self.init_state, self.init_state], axis=0)
        self.attstep_rep = AttentionStep(
                            context=self.context,
                            context_state_size=2*self.state_size,
                            context_mask=self.x_mask,
                            state_size=self.state_size,
                            hidden_size=2*self.state_size,
                            reuse_from=self.attstep)
        self.attstep_rep.context = tf.concat([self.attstep_rep.context,self.attstep_rep.context], axis=1)   
        self.attstep_rep.context_mask = tf.concat([self.attstep_rep.context_mask,self.attstep_rep.context_mask], axis=1)  
        self.attstep_rep.hidden_from_context = tf.concat([self.attstep_rep.hidden_from_context,self.attstep_rep.hidden_from_context], axis=1)  

    def score_true_and_fake(self, config, y, y_mask, fake, fake_mask):
        y = self._pad_and_concat(y, fake)
        y_mask = self._pad_and_concat(y_mask, fake_mask)
        self._repeat_init_state() 
        scores = self.score(config, y, y_mask, use_repeated_context=True)
        true_scores, fake_scores = tf.split(scores, 2, axis=0)
        return true_scores, fake_scores


    def score(self, config, y, y_mask, use_repeated_context=False):
        if use_repeated_context:
            attstep = self.attstep_rep
            init_state = self.init_state_rep
        else:
            attstep = self.attstep
            init_state = self.init_state
        with tf.name_scope("y_embeddings_layer"):
            seq_len = tf.shape(y)[0]
            y_but_last = tf.slice(y, [0,0], [seq_len - 1, -1])
            y_embs = self.y_emb_layer.forward(y_but_last)
            y_embs = tf.pad(y_embs,
                            mode='CONSTANT',
                            paddings=[[1,0],[0,0],[0,0]]) # prepend zeros

        gates_x, proposal_x = self.grustep1.precompute_from_x(y_embs)
        def step_fn(prev_state, x):
            gates_x2d = x[0]
            proposal_x2d = x[1]
            state = self.grustep1.forward(
                        prev_state,
                        gates_x=gates_x2d,
                        proposal_x=proposal_x2d)
            att_ctx = attstep.forward(state) 
            state = self.grustep2.forward(state, att_ctx)
            return state

        self.states = RecurrentLayer(
                    initial_state=init_state,
                    step_fn=step_fn).forward((gates_x, proposal_x))

        y_mask = tf.expand_dims(y_mask, axis=2) # (seqLen, batch, 1)
        self.states = self.states * y_mask
        if not config.use_max_state:
            sum_state = tf.reduce_sum(self.states, axis=0)
            lengths = tf.reduce_sum(y_mask, axis=0)
            mean_state = sum_state / lengths
        else:
            mean_state = tf.reduce_max(self.states, axis=0)

        hidden = mean_state
        for layer in self.hidden_layers:
            hidden = layer.forward(hidden)
        score = self.last_hidden_to_score_layer.forward(hidden)
        score = tf.squeeze(score, axis=1)
        return score
