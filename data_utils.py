from nltk.corpus import brown
import csv
import random
from collections import Counter
import numpy as np


def read_tsv(filename, percentage=1):
    ret = []
    with open(filename, 'rb') as csvfile:
        rowreader = csv.reader(csvfile, delimiter='\t')
        for row in rowreader:
            if random.random() < percentage:
                assert len(row) == 2, "What!!"
                ret.append(tuple(row))
    return ret


class BrownDataset(object):
    # TODO cutoff low frequence OOV
    def __init__(self, training_size=2000, val_size=100):
        self.sentences = []
        self.words = []
        self.unigrams = []
        for sent in brown.sents()[:training_size]:     # TODO remove
            sentence = map(lambda x: x.lower(), sent)
            sentence.insert(0, u'START')
            sentence.append(u'END')
            self.sentences.append(sentence)
            self.words.extend(sentence)
            self.unigrams.extend(sentence)

        self.unigram_freq = dict(Counter(self.unigrams))

        self.num_sentences = len(self.sentences)
        self.words = map(lambda x: x.lower(), self.words)
        self.total_word_cnt = len(self.words) + 2 * len(brown.sents())  # include START and END
        self.words.append(u'OOV')
        self.words.append(u'START')
        self.words.append(u'END')
        self.vocab = set(self.words)

        self.vocab_len = len(self.vocab)
        self.word_to_idx = dict(zip(list(self.vocab), range(self.vocab_len)))
        self.idx_to_word = {v: k for k, v in self.word_to_idx.iteritems()}

        self.training_set = self.sentences

        self.validation_set = []  # TODO sample randomly
        for sent in brown.sents()[training_size:training_size+100]:     # TODO remove
            sentence = map(lambda x: x.lower(), sent)
            sentence.insert(0, u'START')
            sentence.append(u'END')
            self.validation_set.append(sentence)

    def get_batch(self, sent, window_size):
        # <START> and <END> are present
        y = np.array(self.sent_to_idx(sent[1:])) # N x V
        # Add <START>
        [sent.insert(0, u'START') for times in range(window_size-1)]
        sent_idxs = self.sent_to_idx(sent)  # convert to idxs
        X = [sent_idxs[i:i + window_size] for i in range(len(sent_idxs) - window_size)]  # N x h # break into windows

        return X, y

    def sent_to_idx(self, sent):
        return [self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx[u'OOV'] for word in sent]

    def idx_to_sent(self, idxs):
        sent = [self.idx_to_word[idx] for idx in idxs]
        return " ".join(sent)


class SVAgreementCorpus():
    def __init__(self, cut_off=-1):
        self.train_file = './rnn_agr_simple/numpred.train'
        self.val_file = './rnn_agr_simple/numpred.val'
        self.data = []
        self.val_data = []
        self.idx_to_word = {}
        self.word_to_idx = {}
        self.vocab_size = len(self.word_to_idx)

        self.word_to_idx[u'PAD'] = 0
        self.word_to_idx[u'OOV'] = 1
        self.load_data(cut_off)

    def pad_list(self, x, desired_len):
        x.extend([0]*(desired_len-len(x)))

    def get_batch(self, data, size=1):
        batch_mask = np.random.choice(len(data), size)

        #Processing X, y
        batch = data[batch_mask]
        X_batch, y_batch, sents_batch = np.hsplit(batch, 3)
        X_batch = X_batch.flatten()



        y_batch = np.reshape(y_batch == 'VBP', -1)  # 'VBP' = 0, 'VBZ' = 1

        # length of sequences
        lengths = np.vectorize(len)(X_batch)
        max_len = np.max(lengths)

        # padding X_batch
        np.vectorize(self.pad_list)(X_batch, max_len)

        # reverse sorted order
        order = np.flipud(np.argsort(lengths))
        X_batch = X_batch[order]
        y_batch = y_batch[order]
        sents_batch = sents_batch[order]

        # converts numpy 1d array of list to numpy 2d array
        X_batch = np.concatenate(X_batch).reshape(len(X_batch), *np.shape(X_batch[0]))
        y_batch = np.array(y_batch)

        return X_batch, y_batch, sents_batch, lengths[order]


    def __len__(self):
        return len(self.data)

    def load_data(self, cut_off=-1, train_percentage=0.1):
        '''
        train_percentage - what percentage of training data to use (uniformly sampled)
        '''

        assert self.train_file is not None, 'Filename error'

        raw_data = read_tsv(self.train_file, percentage=train_percentage)

        for target, sent in raw_data:
            tokens = sent.split(' ')
            encoded_sent = []
            for token in tokens:
                if token not in self.word_to_idx:
                    self.word_to_idx[token] = len(self.word_to_idx)
                encoded_sent.append(self.word_to_idx[token])
            self.data.append((encoded_sent, target, sent))

        self.idx_to_word = {v: k for k, v in self.word_to_idx.iteritems()}
        self.vocab_size = len(self.word_to_idx)
        self.data = np.array(self.data)

        val_data = read_tsv(self.val_file, percentage=train_percentage)
        for target, sent in val_data:
            tokens = sent.split(' ')
            encoded_sent = [self.word_to_idx[token] if token in self.word_to_idx else self.word_to_idx[u'OOV'] for token in tokens]
            self.val_data.append((encoded_sent, target, sent))
        self.val_data = np.array(self.val_data)

        if cut_off > 0:
            self.data = self.data[:100]
            self.val_data = self.val_data[:100]

if __name__ == '__main__':
    dataset = SVAgreementCorpus(file_path='./rnn_agr_simple/numpred.train')
    print dataset.vocab_size

    # dataset = BrownDataset()
    # print dataset.vocab_len  # 7390
    # print dataset.num_sentences  # 2000
