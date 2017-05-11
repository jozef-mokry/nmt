from tf_layers import *
from util import load_dictionaries, seqs2words
import logging

class CNNCritic(object):
    def __init__(self, config,
                 x, x_mask,
                 y, y_mask,
                 samples, samples_mask):
        

        self.config = config
        logging.info('Creating cnn-critic')
        self.x = x
        self.y = y
        self.x_mask = x_mask
        self.y_mask = y_mask
        self.samples = samples
        self.samples_mask = samples_mask
        self.xT = tf.transpose(x) # ->(batch, seqLen)
        #self.xT = tf.Print(self.xT, [self.xT[0]], summarize=100, message="Prve x")
        self.yT = tf.transpose(y) 
        #self.yT = tf.Print(self.yT, [self.yT[0]], summarize=100, message="Prve y")
        self.samplesT = tf.transpose(samples)
        #self.samplesT = tf.Print(self.samplesT, [self.samplesT[0]], summarize=100, message="Prve samples")
        self.x_maskT = tf.transpose(x_mask)
        #self.x_maskT = tf.Print(self.x_maskT, [self.x_maskT[0]], summarize=100, message="Prve x_maskT")
        self.y_maskT = tf.transpose(y_mask)
        #self.y_maskT = tf.Print(self.y_maskT, [self.y_maskT[0]], summarize=100, message="Prve y_maskT")
        self.samples_maskT = tf.transpose(samples_mask)
        #self.samples_maskT = tf.Print(self.samples_maskT, [self.samples_maskT[0]], summarize=100, message="Prve samples_maskT")

        with tf.name_scope("CriticCNN"):
            self._build_model(config)

    def _build_model(self, config):
        ####### build cnn for x
        with tf.name_scope("x_embeddings_layer"):
            self.x_emb_layer = EmbeddingLayer(
                                vocabulary_size=config.source_vocab_size,
                                embedding_size=config.embedding_size)
            x_embs = self.x_emb_layer.forward(self.xT) #batch, seqLen, embedding
            x_embs *= tf.expand_dims(self.x_maskT, axis=2) #zero-out embeddings for pad symbols
            #x_embs = tf.Print(x_embs, [tf.reduce_sum(x_embs[0], axis=1)], "x_embs", summarize=100)
            #x_embs = tf.Print(x_embs, [x_embs], "x_embs", summarize=1000000)
        with tf.name_scope("x_cnn_layer"):
            self.x_cnn = TextCNNLayer(
                    filter_sizes=config.filter_sizes,
                    filter_counts=config.filter_counts,
                    filter_width=config.embedding_size,
                    activation_fn=tf.nn.relu)
            self.x_features = self.x_cnn.forward(tf.expand_dims(x_embs, axis=3), x_maskT=self.x_maskT) # batch, num_features
            #self.x_features = tf.Print(self.x_features, [tf.reduce_sum(self.x_features, axis=1)], "x_features")

        ###### build features for y
        with tf.name_scope("y_embeddings_layer"):
            self.y_emb_layer = EmbeddingLayer(
                                vocabulary_size=config.target_vocab_size,
                                embedding_size=config.embedding_size)
            y_embs = self.y_emb_layer.forward(self.yT) #seqLen, batch, embedding
            y_embs *= tf.expand_dims(self.y_maskT, axis=2) #zero-out embeddings for pad symbols
            #y_embs = tf.Print(y_embs, [y_embs], "y_embs", summarize=1000000)
        with tf.name_scope("y_cnn_layer"):
            self.y_cnn = TextCNNLayer(
                    filter_sizes=config.filter_sizes,
                    filter_counts=config.filter_counts,
                    filter_width=config.embedding_size,
                    activation_fn=tf.nn.relu)
            self.y_features = self.y_cnn.forward(tf.expand_dims(y_embs, axis=3), x_maskT=self.y_maskT)
            #self.y_features = tf.Print(self.y_features, [tf.reduce_sum(self.y_features, axis=1)], "y_features")

        ##### build features for samples
        with tf.name_scope("y_embeddings_layer"):
            samples_embs = self.y_emb_layer.forward(self.samplesT) #seqLen, batch, embedding
            samples_embs *= tf.expand_dims(self.samples_maskT, axis=2) #zero-out embeddings for pad symbols
        with tf.name_scope("y_cnn_layer"):
            self.samples_features = self.y_cnn.forward(tf.expand_dims(samples_embs, axis=3), x_maskT=self.samples_maskT)
            self.samples_features = tf.Print(self.samples_features, [tf.reduce_sum(self.samples_features, axis=1)], "Samples features")

        ###### compute scores
        xy_features = tf.concat([self.x_features, self.y_features], axis=1) # batch, 2*num_features
        xsamples_features = tf.concat([self.x_features, self.samples_features], axis=1) # batch, 2*num_features
        num_features = sum(config.filter_counts)
        with tf.name_scope("score_layer"):
            self.hidden_layers = []
            for i in range(1, config.num_layers):
                with tf.name_scope("hidden_" + str(i)):
                    logging.debug("Creating hidden layer {}".format(i))
                    layer = FeedForwardLayer(
                                        in_size=2*num_features,
                                        out_size=2*num_features,
                                        non_linearity=tf.nn.tanh)
                    self.hidden_layers.append(layer)
                    xy_features = layer.forward(xy_features)
                    xsamples_features = layer.forward(xsamples_features)
                    xy_features = tf.Print(xy_features, [tf.reduce_sum(xy_features, axis=1)], "xy_features {}".format(i))
                    xsamples_features = tf.Print(xsamples_features, [tf.reduce_sum(xsamples_features, axis=1)], "xsamples_features {}".format(i))
            if config.sigmoid_score:
                non_linearity = tf.nn.sigmoid
            else:
                non_linearity = lambda x: x
            self.features_to_score_layer = FeedForwardLayer(
                                            in_size=2*num_features,
                                            out_size=1,
                                            non_linearity=non_linearity)
            self.true_scores = self.features_to_score_layer.forward(xy_features)
            self.true_scores = tf.squeeze(self.true_scores, axis=1)
            self.fake_scores = self.features_to_score_layer.forward(xsamples_features)
            self.fake_scores = tf.squeeze(self.fake_scores, axis=1)
            #self.true_scores = tf.Print(self.true_scores, [self.true_scores], "true_scores")
            #self.fake_scores = tf.Print(self.fake_scores, [self.fake_scores], "fake_scores")

        ###### compute loss 
        if config.sigmoid_score:
            self.true_loss = tf.log(self.true_scores)
            self.fake_loss = -tf.log(1. - self.fake_scores)
        else:
            self.true_loss = self.true_scores
            self.fake_loss = self.fake_scores
        self.mean_true_loss = tf.reduce_mean(self.true_loss)
        self.mean_fake_loss = tf.reduce_mean(self.fake_loss)
        self.mean_loss = self.mean_fake_loss - self.mean_true_loss

        ##### compute optimizer
        critic_vars = tf.get_collection(
                            key=tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope="CriticCNN")
        if config.use_adam:
            logging.info('Optimizer for critic is Adam')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        else:
            logging.info('Optimizer for critic is RMSprop')
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)

        self.t = tf.Variable(0, name='time', trainable=False, dtype=tf.int32)
        grad_vars = self.optimizer.compute_gradients(
                            self.mean_loss, 
                            var_list=critic_vars)
        if config.clip_c > 0:
            grads, varss = zip(*grad_vars)
            clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_c)
            grad_vars = zip(clipped_grads, varss)
        self.apply_grads = self.optimizer.apply_gradients(grad_vars, global_step=self.t)
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
        _, _, mean_loss, mean_true_loss, mean_fake_loss, samples, samples_mask = sess.run(
                [self.apply_grads,
                 self.clip_vars,
                 self.mean_loss,
                 self.mean_true_loss,
                 self.mean_fake_loss,
                 self.samplesT,
                 self.samples_maskT], inn)

        ### DEBUG ####
       #print 'DEBUG'
       #source = list(x_in.T)
       #target = list(true_in.T)
       #samples = list(samples)
       #samples_mask = list(samples_mask)
       #source_to_num, target_to_num, num_to_source, num_to_target = load_dictionaries(self.config)
       #assert len(source) == len(target) == len(samples) == len(samples_mask)
       #i = 0
       #for src, trg, sample, sample_mask in zip(source, target, samples, samples_mask):
       #    print 
       #    print i, "SRC: {}".format(seqs2words(src, num_to_source))
       #    print i, "TRG: {}".format(seqs2words(trg, num_to_target))
       #    print i, "GEN: {} Length: {}".format(seqs2words(sample, num_to_target), len(sample))
       #    print i, "MSK: {} Length: {}".format(sample_mask, len(sample_mask))
       #    print i, "GENnum: {}".format(sample)
       #    print i, "MSKnum: {}".format(sample_mask)
       #    i+=1

        return mean_loss, mean_true_loss, mean_fake_loss

    def get_true_scores(self):
        return self.true_scores
    def get_fake_scores(self):
        return self.fake_scores
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
