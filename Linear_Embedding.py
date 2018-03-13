import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import argparse

import numpy as np
from data_utils import BrownDataset

class LinearEmbedding(nn.Module):
    def __init__(self, dataset, args, epochs=1, history_len=2, embedding_len=10):
        super(LinearEmbedding, self).__init__()

        self.h = history_len
        self.k = embedding_len
        self.epochs = epochs

        # embedding parameters
        self.word_embeddings = nn.Embedding(dataset.vocab_len, embedding_dim=embedding_len, padding_idx=0)
        self.linear_layer = nn.Linear(self.h*self.k, dataset.vocab_len)

        self.softmax = nn.Softmax()

    def forward(self, X_batch):
        N, h = X_batch.size()

        # input to embeddings
        embedded_X = self.word_embeddings(X_batch)  # N x h x k
        unnorm_probs = self.linear_layer(embedded_X.view(N, -1))
        return unnorm_probs

    def get_sample(self, model, num_samples, dataset):
        # sample
        ret = []
        for i in range(num_samples):
            idxs = dataset.sent_to_idx([u'START' for times in range(self.h)])
            while dataset.idx_to_word[idxs[-1]] != u'END':
                X = Variable(torch.LongTensor([idxs[-self.h:]]), requires_grad=False)
                y_pred = model(X)

                probs = model.softmax(y_pred)
                probs = probs.data.numpy().reshape(-1)
                idx = np.random.choice(dataset.vocab_len, 1, p=probs)[0]
                idxs.append(idx)
            ret.append(dataset.idx_to_sent(idxs))
        return ret


def train(dataset, model, args, debug=True):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])       # TODO add weight_decay

    train_loss = []
    val_loss = []

    for epoch in range(args['epochs']):
        model.train(mode=True)
        epoch_loss = 0
        for itr, sent in enumerate(dataset.sentences):
            optimizer.zero_grad()

            X_batch, y_target = dataset.get_batch(sent, model.h)

            X_batch = Variable(torch.LongTensor(X_batch), requires_grad=False)  # N x max_len
            y_target = Variable(torch.LongTensor(y_target.astype(int)), requires_grad=False)

            y_pred = model.forward(X_batch)

            batch_loss = criterion(y_pred, target=y_target)
            epoch_loss += batch_loss.data.numpy()[0]

            if itr % 200 == 0 and debug:
                print 'iteration=', itr, 'Loss=', batch_loss.data.numpy()[0]

            batch_loss.backward()
            optimizer.step()

        epoch_loss /= len(dataset.sentences)
        print 'epoch=', epoch, 'Loss =', epoch_loss

        # validation loss
        model.train(mode=False)
        avg_val_loss = 0
        for val_itr, val_sent in enumerate(dataset.validation_set):
            val_X, val_y = dataset.get_batch(val_sent, model.h)
            val_X = Variable(torch.LongTensor(val_X), requires_grad=False)  # N x max_len
            val_y = Variable(torch.LongTensor(val_y.astype(int)), requires_grad=False)
            y_pred = model.forward(val_X)

            val_loss = criterion(y_pred, target=val_y)
            avg_val_loss += val_loss.data.numpy()[0]
        avg_val_loss /= len(dataset.validation_set)
        print 'validation loss = ', avg_val_loss

        # save model
        torch.save(model, './save/' + 'Bengio'+str(model.h) + '_' + str(model.k) + '.pt')

        #sample
        for idx, sent in enumerate(model.get_sample(model, 3, dataset)):
            print 'Sent', idx, '"', sent, '"'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linzen et. al.')
    parser.add_argument('-b', '--batch-size', default=20, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')

    args = vars(parser.parse_args())

    dataset = BrownDataset(training_size=2000)
    model = LinearEmbedding(dataset, args, epochs=10, history_len=2, embedding_len=10)
    train(dataset, model, args)