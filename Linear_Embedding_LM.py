import tensorflow as tf
import argparse
import os
import numpy as np
from data_utils import BrownDataset

class LinearEmbeddingLM():
    def __init__(self, dataset, args, epochs=1, history_len=2, embedding_len=10):
        self.h = history_len
        self.k = embedding_len
        self.epochs = epochs

        # input placeholders
        self.input = tf.placeholder(tf.int32, shape=[None, self.h], name='input')       # N x h
        self.y_target = tf.placeholder(tf.float32, shape=[None, dataset.vocab_len], name='output')  # N x V

        # embedding parameters
        self.embedding = tf.Variable(tf.random_normal([dataset.vocab_len, self.k]))

        # input to embeddings
        self.embedded_inp = tf.nn.embedding_lookup(self.embedding, self.input)  # N x h x k
        self.reshaped_inp = tf.reshape(self.embedded_inp, [-1, self.h*self.k])  # N x hk   # concat embd_inp

        # embedding lookup
        self.W = tf.Variable(tf.random_normal([self.h * self.k, dataset.vocab_len]))
        self.b = tf.Variable(tf.random_normal([dataset.vocab_len]))

        self.y_pred = tf.nn.softmax(tf.add(tf.matmul(self.reshaped_inp, self.W), self.b))       # N x V

        # Define loss and optimizer
        self.reg_param = 1e-3
        self.objective = tf.losses.softmax_cross_entropy(self.y_target, self.y_pred) + self.reg_param*tf.nn.l2_loss(self.W)
        self.optimizer = tf.train.AdamOptimizer(args['lr']).minimize(self.objective)

        #model path
        self.model_path = os.path.join('./save', 'LM.ckpt')

    def get_batch(self, sent):
        # <START> and <END> are present
        y = tf.one_hot(dataset.sent_to_idx(sent[1:]), dataset.vocab_len)  # N x V
        # Add <START>
        [sent.insert(0, u'START') for times in range(self.h-1)]
        sent_idxs = dataset.sent_to_idx(sent)  # convert to idxs
        X = [sent_idxs[i:i + self.h] for i in range(len(sent_idxs) - self.h)]  # N x h # break into windows

        return X, y

    def get_sample(self, num_samples, dataset):

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)

            # sample
            ret = []
            for i in range(num_samples):
                idxs = dataset.sent_to_idx([u'START' for times in range(self.h)])
                while dataset.idx_to_word[idxs[-1]] != u'END':
                    out = sess.run([self.y_pred], feed_dict={self.input: [idxs[-self.h:]]})
                    probs = out[0][0]
                    idx = np.random.choice(dataset.vocab_len, 1, p=probs)[0]
                    idxs.append(idx)
                ret.append(dataset.idx_to_sent(idxs))
            return ret

    def train(self, dataset, debug=True):
        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.epochs):
                epoch_loss = 0
                for itr, sent in enumerate(dataset.sentences):
                    X_batch, y_target = self.get_batch(sent)
                    batch_loss, _ = sess.run([self.objective, self.optimizer], feed_dict={self.input: X_batch, self.y_target: y_target.eval()})
                    batch_loss /= len(sent)
                    epoch_loss += batch_loss
                    if itr % 200 == 0 and debug:
                        print 'iteration=', itr, 'Loss=', batch_loss
                epoch_loss /= len(dataset.sentences)
                print 'epoch=', epoch, 'Loss =', epoch_loss

                # validation loss
                avg_val_loss = 0
                for val_itr, val_sent in enumerate(dataset.validation_set):
                    val_inp, val_y = self.get_batch(val_sent)
                    val_loss = sess.run([self.objective], feed_dict={self.input: val_inp, self.y_target: val_y.eval()})
                    val_loss = val_loss[0]
                    val_loss /= len(val_sent)
                    avg_val_loss += val_loss
                avg_val_loss /= len(dataset.validation_set)
                print 'validation loss = ', avg_val_loss

                # save model
                saver = tf.train.Saver()
                saver.save(sess, self.model_path)

                #sample
                for idx, sent in enumerate(self.get_sample(3, dataset)):
                    print 'Sent', idx, '"', sent, '"'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linzen et. al.')
    parser.add_argument('-b', '--batch-size', default=20, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')

    args = vars(parser.parse_args())

    dataset = BrownDataset(training_size=2000)
    model = LinearEmbeddingLM(dataset, args, epochs=10, history_len=2, embedding_len=10)
    model.train(dataset)