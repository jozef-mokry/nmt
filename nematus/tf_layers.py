"""
Layer definitions
"""

from tf_initializers import ortho_weight, norm_weight
import tensorflow as tf
import numpy 
import sys

def matmul3d(x3d, matrix):
    shape = tf.shape(x3d)
    mat_shape = tf.shape(matrix)
    x2d = tf.reshape(x3d, [shape[0]*shape[1], shape[2]])
    result2d = tf.matmul(x2d, matrix)
    result3d = tf.reshape(result2d, [shape[0], shape[1], mat_shape[1]])
    return result3d

class FeedForwardLayer(object):
    def __init__(self,
                 in_size,
                 out_size,
                 non_linearity=tf.nn.tanh):
        self.W = tf.Variable(norm_weight(in_size, out_size), name='W')
        self.b = tf.Variable(numpy.zeros((out_size,)).astype('float32'), name='b')
        self.non_linearity = non_linearity

    def forward(self, x, input_is_3d=False):
        if input_is_3d:
            y = matmul3d(x, self.W) + self.b
        else:
            y = tf.matmul(x, self.W) + self.b
        y = self.non_linearity(y)
        return y

class EmbeddingLayer(object):
    def __init__(self,
                 vocabulary_size,
                 embedding_size):
        self.embeddings = tf.Variable(norm_weight(vocabulary_size, embedding_size),
                                      name='embeddings')
    
    def forward(self, x):
        embs = tf.nn.embedding_lookup(self.embeddings, x)
        return embs

class RecurrentLayer(object):
    def __init__(self,
                 initial_state,
                 step_fn):
        self.initial_state = initial_state
        self.step_fn = step_fn

    def forward(self, x):
        # Assumes that x has shape: time, batch, ...
        states = tf.scan(fn=self.step_fn,
                         elems=x,
                         initializer=self.initial_state)
        return states

class LayerNormLayer(object):
    def __init__(self,
                 layer_size,
                 eps=1e-5):
        new_mean = numpy.zeros(shape=[layer_size], dtype=numpy.float32)
        self.new_mean = tf.Variable(new_mean,
                                    dtype=tf.float32,
                                    name='new_mean')
        new_std = numpy.ones(shape=[layer_size], dtype=numpy.float32)
        self.new_std = tf.Variable(new_std,
                                   dtype=tf.float32,
                                   name='new_std')
        self.eps = eps
    def forward(self, x, input_is_3d=False):
        #NOTE: tf.nn.moments does not support axes=[-1] or axes=[tf.rank(x)-1] :-(
        axis = 2 if input_is_3d else 1
        m, v = tf.nn.moments(x, axes=[axis], keep_dims=True)
        std = tf.sqrt(v + self.eps)
        norm_x = (x-m)/std
        new_x = norm_x*self.new_std + self.new_mean
        return new_x
    
class GRUStep(object):
    def __init__(self, 
                 input_size, 
                 state_size,
                 nematus_compat=False):
        self.state_to_gates = tf.Variable(
                                numpy.concatenate(
                                    [ortho_weight(state_size),
                                     ortho_weight(state_size)],
                                    axis=1), 
                                name='state_to_gates')
        self.input_to_gates = tf.Variable(
                                numpy.concatenate(
                                    [norm_weight(input_size, state_size),
                                     norm_weight(input_size, state_size)],
                                    axis=1), 
                                name='input_to_gates')
        self.gates_bias = tf.Variable(
                            numpy.zeros((2*state_size,)).astype('float32'),
                            name='gates_bias')

        self.state_to_proposal = tf.Variable(
                                    ortho_weight(state_size),
                                    name = 'state_to_proposal')
        self.input_to_proposal = tf.Variable(
                                    norm_weight(input_size, state_size),
                                    name = 'input_to_proposal')
        self.proposal_bias = tf.Variable(
                                    numpy.zeros((state_size,)).astype('float32'),
                                    name='proposal_bias')
        self.nematus_compat = nematus_compat

    

    def _get_gates_x(self, x, input_is_3d=False):
        if input_is_3d:
            gates_x = matmul3d(x, self.input_to_gates) + self.gates_bias
        else:
            gates_x = tf.matmul(x, self.input_to_gates) + self.gates_bias
        return gates_x

    def _get_gates_state(self, prev_state):
        gates_state = tf.matmul(prev_state, self.state_to_gates)
        return gates_state

    def _get_proposal_x(self,x, input_is_3d=False):
        if input_is_3d: 
            proposal_x = matmul3d(x, self.input_to_proposal)
        else:
            proposal_x = tf.matmul(x, self.input_to_proposal)
        if not self.nematus_compat:
            proposal_x += self.proposal_bias
        return proposal_x

    def _get_proposal_state(self, prev_state):
        proposal_state = tf.matmul(prev_state, self.state_to_proposal)
        if self.nematus_compat:
            proposal_state += self.proposal_bias
        return proposal_state

    def precompute_from_x(self, x):
        # compute gates_x and proposal_x in one big matrix multiply
        # if x is fully known upfront 
        # this method exists only for efficiency reasons:W

        gates_x = self._get_gates_x(x, input_is_3d=True)
        proposal_x = self._get_proposal_x(x, input_is_3d=True)
        return gates_x, proposal_x

    def forward(self,
                prev_state,
                x=None,
                gates_x=None,
                gates_state=None,
                proposal_x=None,
                proposal_state=None):
        if gates_x is None:
            gates_x = self._get_gates_x(x) 
        if proposal_x is None:
            proposal_x = self._get_proposal_x(x) 
        if gates_state is None:
            gates_state = self._get_gates_state(prev_state) 
        if proposal_state is None:
            proposal_state = self._get_proposal_state(prev_state) 

        gates = gates_x + gates_state
        gates = tf.nn.sigmoid(gates)
        read_gate, update_gate = tf.split(gates,
                                          num_or_size_splits=2,
                                          axis=1)

        proposal = proposal_state*read_gate + proposal_x
        proposal = tf.tanh(proposal)
        new_state = update_gate*prev_state + (1-update_gate)*proposal

        return new_state

class GRUStepWithNormalization(GRUStep):
    def __init__(self, 
                 input_size, 
                 state_size,
                 nematus_compat=False):
        super(GRUStepWithNormalization, self).__init__(input_size, state_size, nematus_compat)
        #NOTE: GRUStep initializes bias terms which are not used in this class
        #instead norm layers are used
        with tf.name_scope('gates_x_norm'):
            self.gates_x_norm = LayerNormLayer(2*state_size)
        with tf.name_scope('gates_state_norm'):
            self.gates_state_norm = LayerNormLayer(2*state_size)
        with tf.name_scope('proposal_x_norm'):
            self.proposal_x_norm = LayerNormLayer(state_size)
        with tf.name_scope('proposal_state_norm'):
            self.proposal_state_norm = LayerNormLayer(state_size)

    def _get_gates_x(self, x, input_is_3d=False):
        if input_is_3d:
            gates_x = matmul3d(x, self.input_to_gates) 
        else:
            gates_x = tf.matmul(x, self.input_to_gates)
        return self.gates_x_norm.forward(gates_x, input_is_3d=input_is_3d)

    def _get_gates_state(self, prev_state):
        gates_state = tf.matmul(prev_state, self.state_to_gates)
        return self.gates_state_norm.forward(gates_state)

    def _get_proposal_x(self,x, input_is_3d=False):
        if input_is_3d: 
            proposal_x = matmul3d(x, self.input_to_proposal)
        else:
            proposal_x = tf.matmul(x, self.input_to_proposal)
        return self.proposal_x_norm.forward(proposal_x, input_is_3d=input_is_3d)

    def _get_proposal_state(self, prev_state):
        proposal_state = tf.matmul(prev_state, self.state_to_proposal)
        return self.proposal_state_norm.forward(proposal_state)

class AttentionStep(object):
    def __init__(self,
                 context,
                 context_state_size,
                 context_mask,
                 state_size,
                 hidden_size,
                 reuse_from=None):
        if reuse_from is None:
            self.state_to_hidden = tf.Variable(
                                    norm_weight(state_size, hidden_size),
                                    name='state_to_hidden')
            self.context_to_hidden = tf.Variable( #TODO: Nematus uses ortho_weight here - important?
                                        norm_weight(context_state_size, hidden_size), 
                                        name='context_to_hidden')
            self.hidden_bias = tf.Variable(
                                numpy.zeros((hidden_size,)).astype('float32'),
                                name='hidden_bias')
            self.hidden_to_score = tf.Variable(
                                    norm_weight(hidden_size, 1),
                                    name='hidden_to_score')
        else:
            self.state_to_hidden = reuse_from.state_to_hidden
            self.context_to_hidden = reuse_from.context_to_hidden
            self.hidden_bias = reuse_from.hidden_bias
            self.hidden_to_score = reuse_from.hidden_to_score
        
        self.context = context
        self.context_mask = context_mask

        # precompute these activations, they are the same at each step
        # Ideally the compiler would have figured out that too
        self.hidden_from_context = matmul3d(context, self.context_to_hidden)
        self.hidden_from_context += self.hidden_bias

    def forward(self, prev_state):
        hidden = self.hidden_from_context
        hidden += tf.matmul(prev_state, self.state_to_hidden)
        hidden = tf.nn.tanh(hidden)
        # context has shape seqLen x batch x context_state_size
        # mask has shape seqLen x batch

        scores = matmul3d(hidden, self.hidden_to_score) # seqLen x batch x 1
        scores = tf.squeeze(scores, axis=2)
        scores = scores - tf.reduce_max(scores, axis=0, keep_dims=True)
        scores = tf.exp(scores)
        scores *= self.context_mask
        scores = scores / tf.reduce_sum(scores, axis=0, keep_dims=True)

        attention_context = self.context * tf.expand_dims(scores, axis=2)
        attention_context = tf.reduce_sum(attention_context, axis=0, keep_dims=False)

        return attention_context

class Masked_cross_entropy_loss(object):
    def __init__(self,
                 y_true,
                 y_mask):
        self.y_true = y_true
        self.y_mask = y_mask


    def forward(self, logits):
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_true,
                logits=logits)
        #cost has shape seqLen x batch
        cost *= self.y_mask
        cost = tf.reduce_sum(cost, axis=0, keep_dims=False)
        return cost

class TextCNNLayer(object):
    def __init__(self,
                 filter_sizes_and_counts,
                 filter_width,
                 activation_fn):
        self.activation_fn = activation_fn
        self.filters = []
        self.num_features = 0
        num_in_channels = 1
        for num_words, count in filter_sizes_and_counts:
            if count <= 0: 
                print >>sys.stderr, "Filter", num_words, "with zero count -- skipping"
                continue
            self.num_features += count
            num_out_channels = count
            W_init = numpy.random.randn(
                            num_words,
                            filter_width,
                            num_in_channels,
                            num_out_channels)
            W = tf.Variable(W_init, 
                            dtype=tf.float32,
                            name='W-{}x{}'.format(num_words, num_out_channels))
            self.filters.append(W)
        b_init = numpy.zeros((self.num_features,))
        self.b = tf.Variable(b_init, dtype=tf.float32, name='b')

    def forward(self, x):
        conv_outs = []
        for filter_ in self.filters:
            out = tf.nn.conv2d(
                    input=x,
                    filter=filter_,
                    strides=[1,1,1,1],
                    padding='VALID') # out.shape=(batch, seqLen-num_words+1, 1, count)
            out = tf.squeeze(out, axis=2)
            out = tf.reduce_max(out, axis=1) # shape (batch, count)
            conv_outs.append(out)
        features = tf.concat(conv_outs, axis=1) # shape (batch, num_features)
        features += self.b
        features = self.activation_fn(features)
        return features

def create_samples_mask(samples, pad_to_len=None):
    lengths = tf.reduce_sum(
                tf.cast(tf.not_equal(samples, 0), dtype=tf.float32),
                axis=0)
    if pad_to_len is None:
        lengths = tf.where(tf.equal(samples[-1], 0), lengths + 1, lengths) #Add 1 for eos if reached
        samples_mask = tf.transpose(tf.sequence_mask(lengths, dtype=tf.float32)) 
    else:
        print 'Lengths will be at least of size', pad_to_len
        lengths = tf.where(tf.equal(samples[-1], 0), lengths + 1, lengths) #Add 1 for eos if reached
        maxlen = tf.maximum(tf.reduce_max(lengths), pad_to_len)
        samples_mask = tf.transpose(tf.sequence_mask(lengths, dtype=tf.float32, maxlen=maxlen)) 
    return samples_mask

def pad_to_at_least_len(x, pad_to_len):
    seqLen = tf.shape(x)[0]

    x_pad = tf.cond(seqLen < pad_to_len,
                    lambda: tf.pad(x, [[0, pad_to_len-seqLen],[0,0]]),
                    lambda: x)
    return x_pad

    

