import numpy

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20,
                 use_train_weights=False):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.weight_buffer = []
        self.use_train_weights = use_train_weights
        self.k = batch_size * maxibatch_size
        

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        weights = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        if self.use_train_weights:
            assert len(self.source_buffer) == len(self.weight_buffer), 'Buffer size mismatch! 2'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                tt = self.target.readline()

                if self.use_train_weights:
                    ww_s, ss = ss.split(" ||| ", 1)
                    ww_t, tt = tt.split(" ||| ", 1)
                    assert ww_s == ww_t, 'Weights in source and target file do not agree!'

                    ww = ww_s

                ss = ss.split()
                tt = tt.split()
                
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if self.use_train_weights:
                    self.weight_buffer.append(ww)

                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

                if self.use_train_weights:
                    _wbuf = [self.weight_buffer[i] for i in tidx]
                    self.weight_buffer = _wbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
                if self.use_train_weights:
                    self.weight_buffer.reverse()


        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    tmp.append(w)
                ss = tmp

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                source.append(ss)
                target.append(tt)

                if self.use_train_weights:
                    # read from the weight buffer
                    ww = float(self.weight_buffer.pop())
                    weights.append(ww)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if self.use_train_weights:
            return source, target, weights
        else:
            return source, target, [1]*len(source)
