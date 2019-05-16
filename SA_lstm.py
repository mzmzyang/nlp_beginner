import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.functional as F


def vocab_list(train_data, vali_data, test_data):
    word_list = []
    train_words = []
    vali_words = []
    test_words = []
    for phrase in train_data:
        t = []
        for word in phrase.split(' '):
            word_list.append(word.lower())
            t.append(word.lower())
        train_words.append(t)
    for phrase in vali_data:
        t = []
        for word in phrase.split(' '):
            word_list.append(word.lower())
            t.append(word.lower())
        vali_words.append(t)
    for phrase in test_data:
        t = []
        for word in phrase.split(' '):
            t.append(word.lower())
        test_words.append(t)
    return list(set(word_list)), train_words, vali_words, test_words


def transition(word_list):
    word2idx = {word: idx + 1 for idx, word in enumerate(word_list)}
    idx2word = {idx + 1: word for idx, word in enumerate(word_list)}
    return word2idx, idx2word


def encode_phrase(data, word2idx):
    encode_data = []
    for phrase in data:
        t = []
        for word in phrase:
            if word in word2idx:
                t.append(word2idx[word])
            else:
                t.append(0)
        encode_data.append(t)
    return encode_data


def pad_phrase(encode_data, max_length):
    pad_encode_data = []
    for phrase in encode_data:
        temp_phrase = phrase
        if len(temp_phrase) > max_length:
            pad_encode_data.append(temp_phrase[:max_length])
        else:
            while len(temp_phrase) < max_length:
                temp_phrase.append(0)
            pad_encode_data.append(temp_phrase)
    return pad_encode_data


def precess_dataset(max_length):
    train_dataset = pd.read_csv("./dataset/train.tsv", sep='\t')
    test_dataset = pd.read_csv("./dataset/test.tsv", sep='\t')
    # print(train_dataset.info())
    train_phrase = train_dataset['Phrase']
    test_phrase = test_dataset['Phrase']
    train_y = train_dataset['Sentiment'].values

    vali_phrase = train_phrase[120000:]
    train_phrase = train_phrase[:120000]
    vali_y = train_y[120000:]
    train_y = train_y[:120000]

    word_list, train_words, vali_words, test_words = vocab_list(train_phrase, vali_phrase, test_phrase)
    word_size = len(word_list) + 1
    word2idx, idx2word = transition(word_list)

    train_x = pad_phrase(encode_phrase(train_words, word2idx), max_length)
    vali_x = pad_phrase(encode_phrase(vali_words, word2idx), max_length)
    test_x = pad_phrase(encode_phrase(test_words, word2idx), max_length)

    return word_size + 1, word2idx, idx2word, train_x, vali_x, train_y, vali_y, test_x


class MySA(nn.Module):
    def __init__(self, vocb_size, emd_dim, hidden_size, layer_num, class_size):
        super(MySA, self).__init__()
        self.embedding = nn.Embedding(vocb_size, emd_dim)
        self.myLSTM = nn.LSTM(emd_dim, hidden_size, layer_num, bidirectional=True)
        self.liner = nn.Linear(hidden_size * 4, 20)
        self.dropout = nn.Dropout(0.5)
        self.predict = nn.Linear(20, class_size)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        status, hidden = self.myLSTM(embed.permute(1, 0, 2))
        encode = torch.cat((status[0], status[-1]), dim=1)
        out = self.liner(encode)
        out = self.dropout(out)
        out = self.predict(out)
        return out


LR = 0.01
EPOCH = 5
MAX_LENGTH = 25
BATCH_SIZE = 128

vocb_size, word2idx, idx2word, train_x, vali_x, train_y, vali_y, test_x = precess_dataset(MAX_LENGTH)
train_x = torch.LongTensor(train_x)
train_y = torch.tensor(train_y)
vali_x = torch.LongTensor(vali_x)
vali_y = torch.tensor(vali_y)
test_x = torch.LongTensor(test_x)
train_set = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

mySA = MySA(vocb_size, 20, 20, 1, 5)
print(mySA)
optimizer = torch.optim.Adam(mySA.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
mySA.cuda()
loss_func.cuda()

for epoch in range(EPOCH):
    for idx, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        output = mySA(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            mySA.eval()
            vali_x = vali_x.cuda()
            vali_y = vali_y.cuda()
            vali_output = mySA(vali_x)
            pred_output = torch.max(vali_output, dim=1)[1]
            accuracy = float(torch.sum(pred_output == vali_y)) / float(vali_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| validation accuracy: %.4f' % accuracy)
            mySA.train()

mySA.eval()
test_x = test_x.cuda()
pred_test = mySA(test_x)
pred_output = torch.max(pred_test, dim=1)[1]
pred_output = pred_output.type(torch.int32)

sub_file = pd.read_csv("./dataset/test.tsv", sep='\t')
print(sub_file.info())
del sub_file['SentenceId']
del sub_file['Phrase']
print(sub_file.info())
sub_file['Sentiment'] = pred_output.cpu()
sub_file.to_csv("./dataset/submission.csv", index=False)
print(sub_file.info())
