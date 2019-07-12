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


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hid_dim, n_heads):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.num_heads = self.hid_dim // self.n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(input_dim, hid_dim)
        self.w_k = nn.Linear(input_dim, hid_dim)
        self.w_v = nn.Linear(input_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value):
        bsz = query.shape[0]
        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.num_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.num_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.num_heads).permute(0, 2, 1, 3)
        # Q, K, V = [batch size, n heads, sent len, num_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, sent len, sent len]

        attention = F.softmax(energy, dim=-1)
        # attention = [batch size, n heads, sent len, sent len]

        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len, num_heads]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len, n heads, num_heads]

        x = x.view(bsz, -1, self.n_heads * self.num_heads)
        # x = [batch size, src sent len, hid dim]

        x = self.fc(x)
        # x = [batch size, sent len, hid dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, self_attention, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, hid_dim, n_heads)
        self.do = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src sent len, hid dim]
        src = self.ln(src + self.do(self.sa(src, src, src)))
        return src


class Transformer(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, encoder_layer, self_attention,
                 dropout, glove, class_size, max_length):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.dropout = dropout
        self.tok_embedding = nn.Embedding.from_pretrained(glove, freeze=False)

        self.layers = nn.ModuleList(
            [encoder_layer(hid_dim, n_heads, self_attention, dropout)
             for i in range(n_layers)])

        self.do = nn.Dropout(dropout)
        self.proj = nn.Linear(max_length, 1)
        self.predict = nn.Linear(hid_dim, class_size)

    def forward(self, src):
        # src = [batch size, src sent len]
        src = self.tok_embedding(src)
        # src = [batch size, src sent len, hid dim]

        for layer in self.layers:
            src = layer(src)

        src = src.permute(0, 2, 1)
        src = self.proj(src)
        src = src.squeeze(2)
        return self.predict(src)


LR = 0.01
EPOCH = 3
MAX_LENGTH = 10
BATCH_SIZE = 16

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

model = Transformer(input_dim=vocb_size,
                    hid_dim=50,
                    n_layers=5,
                    n_heads=5,
                    encoder_layer=EncoderLayer,
                    self_attention=SelfAttention,
                    dropout=0.1,
                    glove=glove,
                    class_size=5,
                    max_length=MAX_LENGTH)

optimizer = torch.optim.SGD(model.parameters(), lr=LR)
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
