import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F


from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import nltk
from nltk import bigrams, trigrams

from collections import Counter
import re

#--------------------Sentiment Model Config----------
#Python version: 3.7.*
#Pytorch Version: 1.4 or higher
#Model: "sentiment_model_ngram.pt"
#Tokenizer: "tokenizer2_sent_attention_ngram.pickle"
#number of Vocab: 1000
#sentence Length: 100
#Embedding Dim: 100
#--------------------Model Config END-----------------

# device = ("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class AttentionModel(nn.Module):
    def __init__(self,seq_len,embd_dim,num_vocab):
        super(AttentionModel,self).__init__()
        self.seq_len = seq_len
        self.embd_dim = embd_dim
        self.num_vocab = num_vocab
        self.embd = nn.Embedding(num_vocab,embd_dim,padding_idx=0)
        self.transformer = nn.TransformerEncoderLayer(embd_dim,nhead=4)
        self.fc = nn.Linear(seq_len*embd_dim,1024)
        self.out = nn.Linear(1024,3)
    def forward(self,X):
        em_out = self.embd(X)
        tr_in = em_out.reshape(self.seq_len,-1,self.embd_dim)
        tr_out = self.transformer(tr_in)
        x = torch.mul(em_out,tr_out.reshape(-1,self.seq_len,self.embd_dim))
        x = x.reshape(-1,self.seq_len*self.embd_dim)
        x = self.fc(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x)
        return x


def get_n_gram(string):
    tri_str = []
    if len(string)<3:
        tri_str.append(string)
    else:
        for word in string.split(" "):
            if len(word)<3:
                tri_str.append(word)
            else:
                string_bigrams = trigrams(word)
                r = ["".join(w) for w in string_bigrams]
                r = " ".join(r)
                tri_str.append(r)
    return " ".join(tri_str)


def clean_dataset(texts: list, max_sq_len=100, tok=None, ngram=True, num_voc=1000):
    clean_texts = []
    stop_words = ['i', 'am', 'an', 'the', 'is', 'are', 'we', 'you', 'apni', 'ami', 'ki', 'keno', 'kno', 'ache', 'ase',
                  'ace', 'to', 'bhai',
                  'bai', 'love', 'lab', 'naki', 'nki', 'yo', 'bro', 'mia', 'k', 'n', 'and', 'xoxo', 'for', 'as', 'a',
                  'korbo', 'kori', 'if',
                  'kisu', 'kicu', 'dhk', 'this', 'they', 'ora', 'gp', 'robi', 'bl', 'foodpanda', 'hungrynaki', 'shohoz',
                  'nogod', 'bkash',
                  'rocket', 'uber', 'pathao', 'shohoz', 'banglaflix', 'er', 'e']
    for text in texts:
        text = " ".join([w for w in text.split('/') if w not in stop_words])
        s = re.sub("[._<0-9(#)-@]", "", str(text).lower())
        s = [w for w in s.split(" ") if w not in stop_words]
        s = " ".join(s)
        clean_texts.append(s)

    if ngram:
        clean_texts = [get_n_gram(sent) for sent in clean_texts]

    if tok:
        texts_seq = tok.texts_to_sequences(clean_texts)
    else:
        tok = Tokenizer(num_words=num_voc)
        tok.fit_on_texts(clean_texts)
        texts_seq = tok.texts_to_sequences(clean_texts)

    padded_seq = pad_sequences(texts_seq, maxlen=max_sq_len, padding='post', truncating='post')
    return padded_seq, clean_texts, tok

def infer_sentiment(s, tok, model):
    flag_ = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}

    padded_data, ct, t = clean_dataset([s], 100, tok, ngram=True)
    tensor = torch.from_numpy(padded_data).type(torch.LongTensor)
    print(device)
    tensor = tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        cls = torch.argmax(logits, 1).item()
        return (flag_[cls], logits.cpu().numpy())