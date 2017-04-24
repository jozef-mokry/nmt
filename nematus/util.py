'''
Utility functions
'''

import sys
import json
import cPickle as pkl
import numpy
import data_iterator as di
import os
import logging


def initial_setup(config):
    translating = config.translate_valid
    validating = config.run_validation
    training = not (translating or validating)

    if training:
        create_save_dir(config)
    setup_logging(config, training, True)

def create_save_dir(config):
    path = os.path.dirname(os.path.abspath(config.saveto))
    os.mkdir(path)

def setup_logging(config, log_to_files, format_logs):
    if format_logs:
        format_string = '[%(asctime)s %(levelname)-5s] %(message)s'
    else:
        format_string = '%(message)s'
    formatter = logging.Formatter(format_string)

    logging.basicConfig(level=logging.DEBUG,
                        format=format_string,
                        stream=sys.stderr)

    if log_to_files:
        path = os.path.dirname(os.path.abspath(config.saveto))
        info_file = os.path.join(path, 'info.log')
        debug_file = os.path.join(path, 'debug.log') 
        # save INFO messages into a file
        f_out = logging.FileHandler(filename=info_file, mode='a', delay=True)
        f_out.setLevel(logging.DEBUG)
        f_out.setFormatter(formatter)
        logging.getLogger('').addHandler(f_out)

        # save DEBUG messages into a file
        #f_err = logging.FileHandler(filename=debug_file, mode='a', delay=True)
        #f_err.setLevel(logging.DEBUG)
        #f_err.setFormatter(formatter)
        #logging.getLogger('').addHandler(f_err)

        def log_uncaught_exception(exc_type, exc_val, exc_trace):
            logging.error("Uncaught exception", exc_info=(exc_type, exc_val, exc_trace))

        # handle uncaught exception
        sys.excepthook = log_uncaught_exception

def load_data(config):
    logging.info('Reading data...')
    text_iterator = di.TextIterator(
                        source=config.source_dataset,
                        target=config.target_dataset,
                        source_dicts=[config.source_vocab],
                        target_dict=config.target_vocab,
                        batch_size=config.batch_size,
                        maxlen=config.maxlen,
                        n_words_source=config.source_vocab_size,
                        n_words_target=config.target_vocab_size,
                        skip_empty=True,
                        shuffle_each_epoch=config.shuffle_each_epoch,
                        sort_by_length=config.sort_by_length,
                        maxibatch_size=config.maxibatch_size,
                        keep_data_in_memory=config.keep_train_set_in_memory)

    if config.validFreq:
        valid_text_iterator = di.TextIterator(
                                source=config.valid_source_dataset,
                                target=config.valid_target_dataset,
                                source_dicts=[config.source_vocab],
                                target_dict=config.target_vocab,
                                batch_size=config.valid_batch_size,
                                maxlen=config.validation_maxlen,
                                n_words_source=config.source_vocab_size,
                                n_words_target=config.target_vocab_size,
                                shuffle_each_epoch=False,
                                sort_by_length=True,
                                maxibatch_size=config.maxibatch_size)
    else:
        valid_text_iterator = None
    logging.info('Done')
    return text_iterator, valid_text_iterator

def load_dictionaries(config):
    source_to_num = load_dict(config.source_vocab)
    target_to_num = load_dict(config.target_vocab)
    num_to_source = reverse_dict(source_to_num)
    num_to_target = reverse_dict(target_to_num)
    return source_to_num, target_to_num, num_to_source, num_to_target

# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    n_factors = len(seqs_x[0][0])
    assert n_factors == 1
    maxlen_x = numpy.max(lengths_x) + 1 
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    # there is only one factor, get rid of that dimension
    x = x.squeeze(axis=0)

    return x, x_mask, y, y_mask

def merge_batches(true_in, true_mask_in, fake_in, fake_mask_in):
    true_len, n_true = true_in.shape
    fake_len, n_fake = fake_in.shape
    assert n_true == n_fake, (n_true, n_fake)
    both_len, n_both = max(true_len, fake_len), (n_true+n_fake)
    both_in = numpy.zeros((both_len, n_both), dtype=numpy.int32)
    both_mask_in = numpy.zeros((both_len, n_both))
    both_in[:true_len, :n_true] = true_in
    both_in[:fake_len, n_true:] = fake_in
    both_mask_in[:true_len, :n_true] = true_mask_in
    both_mask_in[:fake_len, n_true:] = fake_mask_in
    labels_in = numpy.ones((n_true+n_fake,), dtype=numpy.float32)
    labels_in[n_true:] = -1
    return both_in, both_mask_in, labels_in

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def seqs2words(seq, inverse_target_dictionary):
    words = []
    for i, w in enumerate(seq):
        if w == 0:
            assert (i == len(seq) - 1) or (seq[i+1] == 0), ('Zero not at the end of sequence', seq)
        elif w in inverse_target_dictionary:
            words.append(inverse_target_dictionary[w])
        else:
            words.append('UNK')
    return ' '.join(words)

def reverse_dict(dictt):
    keys, values = zip(*dictt.items())
    r_dictt = dict(zip(values, keys))
    return r_dictt
