import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


def vocab_list(phrase_data, word_set, is_test):
    phrase_list = []
    for phrase in phrase_data:
        phrase = phrase.lower()
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'d", " wound", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"[^a-z]", " ", phrase)
        temp = []
        for word in phrase.split(' '):
            temp.append(word)
            if not is_test:
                word_set.add(word)
        phrase_list.append(temp)

    return phrase_list


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


def get_glove(words_set):
    glove = torch.zeros([len(words_set) + 1, 50])
    word2idx = {}
    word2idx['<unk>'] = 0
    idx = 1
    with open("./glove/glove.6B.50d.txt") as glove_file:
        for line in glove_file:
            temp = line.split()
            if temp[0] in words_set:
                glove[idx] = torch.from_numpy(np.array(temp[1:]).astype(np.float))
                word2idx[temp[0]] = idx
                idx = idx + 1
    return word2idx, glove[:idx, :]


def precess_dataset(max_length):
    train_dataset = pd.read_csv("./dataset/train.tsv", sep='\t')
    test_dataset = pd.read_csv("./dataset/test.tsv", sep='\t')
    # print(train_dataset.info())

    train_phrase = train_dataset['Phrase']
    test_phrase = test_dataset['Phrase']
    train_y = train_dataset['Sentiment'].values

    validation_phrase = train_phrase[120000:]
    train_phrase = train_phrase[:120000]
    validation_y = train_y[120000:]
    train_y = train_y[:120000]

    word_set = set()

    train_words = vocab_list(train_phrase, word_set, False)
    validation_words = vocab_list(validation_phrase, word_set, False)
    test_words = vocab_list(test_phrase, word_set, True)

    word2idx, glove = get_glove(word_set)

    train_x = pad_phrase(encode_phrase(train_words, word2idx), max_length)
    validation_x = pad_phrase(encode_phrase(validation_words, word2idx), max_length)
    test_x = pad_phrase(encode_phrase(test_words, word2idx), max_length)

    return len(word2idx), word2idx, glove, train_x, validation_x, train_y, validation_y, test_x


class myGloVeCNN(nn.Module):
    def __init__(self, vocb_size, emd_dim, seq_len, dropout, class_size, glove):
        super(myGloVeCNN, self).__init__()

        self.embedding = nn.Embedding(vocb_size, emd_dim)
        # self.embedding = nn.Embedding.from_pretrained(glove, freeze=False)
        self.conv1 = nn.Conv2d(1, 1, (3, emd_dim))
        self.conv2 = nn.Conv2d(1, 1, (4, emd_dim))
        self.conv3 = nn.Conv2d(1, 1, (5, emd_dim))
        self.conv4 = nn.Conv2d(1, 1, (6, emd_dim))
        self.conv5 = nn.Conv2d(1, 1, (7, emd_dim))
        self.conv6 = nn.Conv2d(1, 1, (8, emd_dim))

        self.pool1 = nn.MaxPool2d((seq_len - 2, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 3, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 4, 1))
        self.pool4 = nn.MaxPool2d((seq_len - 5, 1))
        self.pool5 = nn.MaxPool2d((seq_len - 6, 1))
        self.pool6 = nn.MaxPool2d((seq_len - 7, 1))

        self.liner = nn.Linear(6, class_size)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        embed = embed.view(inputs.shape[0], 1, inputs.shape[1], -1)
        x1 = F.relu(self.conv1(embed))
        x2 = F.relu(self.conv2(embed))
        x3 = F.relu(self.conv3(embed))
        x4 = F.relu(self.conv4(embed))
        x5 = F.relu(self.conv5(embed))
        x6 = F.relu(self.conv6(embed))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        x4 = self.pool4(x4)
        x5 = self.pool5(x5)
        x6 = self.pool6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6), -1)
        x = x.view(inputs.shape[0], -1)

        out = self.liner(x)
        return out


LR = 0.01
EPOCH = 3
MAX_LENGTH = 36
BATCH_SIZE = 128

vocb_size, word2idx, glove, train_x, validation_x, train_y, validation_y, test_x = precess_dataset(MAX_LENGTH)
train_x = torch.LongTensor(train_x)
train_y = torch.tensor(train_y)
validation_x = torch.LongTensor(validation_x)
validation_y = torch.tensor(validation_y)
test_x = torch.LongTensor(test_x)

train_set = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=train_set,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

model = myGloVeCNN(vocb_size=vocb_size,
            emd_dim=50,
            seq_len=MAX_LENGTH,
            dropout=0.1,
            class_size=5,
            glove=glove)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
print(model)
model.cuda()
loss_func.cuda()

for epoch in range(EPOCH):
    for idx, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()

        pred_train = model(x)
        loss = loss_func(pred_train, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            model.eval()
            validation_x = validation_x.cuda()
            validation_y = validation_y.cuda()

            pred_validation = model(validation_x)
            pred_output = torch.max(pred_validation, dim=1)[1]
            accuracy = float(torch.sum(pred_output == validation_y)) / float(validation_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| validation accuracy: %.4f' % accuracy)
            model.train()

model.eval()
test_x = test_x.cuda()
pred_test = model(test_x)

pred_output = torch.max(pred_test, dim=1)[1]
pred_output = pred_output.type(torch.int32)

submit_file = pd.read_csv("./dataset/test.tsv", sep='\t')
del submit_file['SentenceId']
del submit_file['Phrase']

submit_file['Sentiment'] = pred_output.cpu()
submit_file.to_csv("./dataset/submission.csv", index=False)
print(submit_file.info())
