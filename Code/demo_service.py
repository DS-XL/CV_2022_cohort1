'''
encoding: utf-8
Author : Jingxuan Li
Datetime : 2022/7/23 21:33
Product : PyCharm
File : demo_service_02.py
Description : 文件说明
'''
import json

from flask import Flask
from flask import request
from flask import Response
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from collections import Counter
import time
import random
import os
import sys
import pathlib
import jieba
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

__APP__ = Flask(__name__)
__PREDICTOR__ = None


class config:
  # General
  hidden_size: int = 256
  dec_hidden_size: Optional[int] = 256
  embed_size: int = 256

  # Data
  max_vocab_size = 50000
  embed_file: Optional[str] = None  # use pre-trained embeddings

  # working directory
  root_path = os.path.abspath(os.path.dirname('D:/0 - Google Drive/OneCareer/CV Advance/Data/'))
  os.chdir(root_path)

  data_path = os.path.join(root_path, 'data_pairs.txt')
  train_data_path = os.path.join(root_path, 'train.txt')
  val_data_path: Optional[str] = os.path.join(root_path, 'dev.txt')
  test_data_path: Optional[str] = os.path.join(root_path, 'test.txt')
  stop_word_file = os.path.join(root_path, 'HIT_stop_words.txt')

  max_src_len: int = 300  # exclusive of special tokens such as EOS
  max_tgt_len: int = 100  # exclusive of special tokens such as EOS
  truncate_src: bool = True
  truncate_tgt: bool = True
  min_dec_steps: int = 20
  max_dec_steps: int = 50
  enc_rnn_dropout: float = 0.5
  enc_attn: bool = True
  dec_attn: bool = True
  dec_in_dropout = 0.3
  dec_rnn_dropout = 0.3
  dec_out_dropout = 0.3


  # Training
  trunc_norm_init_std = 1e-4
  eps = 1e-31
  learning_rate = 0.0001
  lr_decay = 0.0
  initial_accumulator_value = 0.1
  epochs = 10
  batch_size = 6

  pointer = True
  coverage = True
  #fine_tune = False
  scheduled_sampling = False # Teacher forcing
  weight_tying = False
  max_grad_norm = 10.0
  is_cuda = False
  DEVICE = torch.device("cuda" if is_cuda else "cpu")
  LAMBDA = 1


  # Beam search
  beam_size: int = 3
  alpha = 0.2
  beta = 0.2
  gamma = 2000

  # Model save path
  model_name = 'baseline'
  if pointer:
    model_name = 'pgn'
  model_name += '_' + str(hidden_size)
  if coverage:
    model_name += '_cov'
  if scheduled_sampling:
    model_name += '_teacher'

  #调用系统命令行来创建文件夹
  directory_path = os.path.join(root_path, 'model_save/' + model_name + '/')
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)

  encoder_save_name = os.path.join(root_path, 'model_save/' + model_name + '/encoder.pt')
  decoder_save_name = os.path.join(root_path, 'model_save/' + model_name + '/decoder.pt')
  attention_save_name = os.path.join(root_path, 'model_save/' + model_name + '/attention.pt')
  reduce_state_save_name = os.path.join(root_path, 'model_save/' + model_name + '/reduce_state.pt')
  losses_path = os.path.join(root_path, 'model_save/' + model_name + '/validation_loss.pkl')


class Vocab:
    # assign initial index to pad, sos(indicate the start of sequence), eos(indicate the end of sequence) and unk (unknown token)
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

    def addWord(self, words):
        """add new token to vocab and map word to index
        """
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def __len__(self):
        return len(self.index2word)

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def size(self):
        """Returns the total size of the vocabulary
        """
        return len(self.index2word)

def src2ids(src_list, vocab):
  # oov: out of vocabulary
  ids = []
  oovs = []
  unk_id = vocab.UNK
  for w in src_list:
    i = vocab[w]
    if i == unk_id:
      if w not in oovs:
        oovs.append(w)
      oov_num = oovs.index(w)
      #print('oov_num is :', oov_num)
      ids.append(vocab.size() + oov_num)
    else:
      ids.append(i)
  return ids, oovs

class PairDataset:
    """create src-tgt pairs dataset for model.
    """

    def __init__(self, filename):
        self.filename = filename
        self.pairs = []

        with open(filename, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # split the source and target by the <sep> tag.
                pair = line.strip().split('<sep>')
                # split lines to tokens in src and tgt
                src = pair[1].split()
                tgt = pair[2].split()
                self.pairs.append((src, tgt))
        print("%d pairs." % len(self.pairs))

    def build_vocab(self, embed_file: str = None) -> Vocab:
        # count word frequency in src-tgt pairs
        word_counts = Counter()
        text = [src + tgt for src, tgt in self.pairs]
        for sentence in text:
            for token in sentence:
                word_counts[token] += 1

        # create word to index with the most frequent words having smaller index
        vocab = Vocab()
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.addWord([word])
        # if embed_file is not None:
        # count = vocab.load_embeddings(embed_file)
        # print("%d pre-trained embeddings loaded." % count)

        return vocab

def timer(module):
    def wrapper(func):
        def cal_time(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper

def outputids2words(id_list, source_oovs, vocab):
    """
        Maps output ids to words.
    """
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]  # might be [UNK]
        except IndexError:  # w is OOV
            assert_msg = "Error: cannot find the ID the in the vocabulary."
            assert source_oovs is not None, assert_msg
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:  # i doesn't correspond to an source oov
                raise ValueError(
                    'Error: model produced word ID %i corresponding to source OOV %i \
                     but this example only has %i source OOVs'
                    % (i, source_oov_idx, len(source_oovs)))
        words.append(w)
    return ' '.join(words)

def replace_oovs(in_tensor, vocab):
    """Replace oov tokens in a tensor with the <UNK> token.
    """
    oov_token = torch.full(in_tensor.shape, vocab.UNK).long().to(config.DEVICE)
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor

class Encoder(nn.Module):
  """encoder part of Seq2Seq model"""
  def __init__(self, vocab_size, embed_size, hidden_size, rnn_drop: float = 0):
    super(Encoder, self).__init__()

    # embedding layer
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.hidden_size = hidden_size

    # define bidirectional lstm
    self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, dropout=rnn_drop, batch_first=True)

    # since bidirectiona is true, the dimension of output will double
    #self.fc = nn.Linear(hidden_size * 2, dec_hidden_size)

  def forward(self, x):
    """forward propagation for encoder"""
    embedded = self.embedding(x)
    output, hidden = self.lstm(embedded)
    return output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size=None, is_cuda=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, x_t, decoder_states, context_vector):
        """Define forward propagation for the decoder.
        """
        decoder_emb = self.embedding(x_t)
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # concatenate context vector and decoder state
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)

        # calculate vocabulary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)
        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        return p_vocab, decoder_states

class Attention(nn.Module):
  """attention part"""
  def __init__(self, hidden_units):
    super(Attention, self).__init__()
    # Define feed-forward layers.
    self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
    self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)
    self.v = nn.Linear(2*hidden_units, 1, bias=False)

    #self.wc = nn.Linear(1, 2*hidden_units, bias=False)

  # def forward(self, decoder_states, encoder_output, x_padding_masks):
  def forward(self, decoder_states, encoder_output, x_padding_masks):
    """Define forward propagation for the attention network.
    """
    # Concatenate h and c to get s_t and expand the dim of s_t.
    h_dec, c_dec = decoder_states
    # (1, batch_size, 2*hidden_units)
    s_t = torch.cat([h_dec, c_dec], dim=2)
    # (batch_size, 1, 2*hidden_units)
    s_t = s_t.transpose(0, 1)
    # (batch_size, seq_length, 2*hidden_units)
    s_t = s_t.expand_as(encoder_output).contiguous()

    # calculate attention scores
    # Wh h_* (batch_size, seq_length, 2*hidden_units)
    encoder_features = self.Wh(encoder_output.contiguous())
    # Ws s_t (batch_size, seq_length, 2*hidden_units)
    decoder_features = self.Ws(s_t)
    # (batch_size, seq_length, 2*hidden_units)
    att_inputs = encoder_features + decoder_features

    # (batch_size, seq_length, 1)
    score = self.v(torch.tanh(att_inputs))
    # (batch_size, seq_length)
    attention_weights = F.softmax(score, dim=1).squeeze(2)
    attention_weights = attention_weights * x_padding_masks
    # Normalize attention weights after excluding padded positions.
    normalization_factor = attention_weights.sum(1, keepdim=True)
    attention_weights = attention_weights / normalization_factor
    # (batch_size, 1, 2*hidden_units)
    context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output)
    # (batch_size, 2*hidden_units)
    context_vector = context_vector.squeeze(1)

    return context_vector, attention_weights

class ReduceState(nn.Module):
    """because we use BiLSTM in encoder, and LSTM in decoder,
    when we use output from encoder as initialized hidden state for decoder, we need to reduce state for encoder
    """
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        """The forward propagation of reduce state module.
        """
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)


class Seq2seq(nn.Module):
    def __init__(self, v):
        super(Seq2seq, self).__init__()
        self.v = v
        self.DEVICE = config.DEVICE
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(len(v), config.embed_size, config.hidden_size, )
        self.decoder = Decoder(len(v), config.embed_size, config.hidden_size, )
        self.reduce_state = ReduceState()

    def forward(self, x, x_len, y, len_oovs, batch):
        x_copy = replace_oovs(x, self.v)
        x_padding_masks = torch.ne(x, 0).byte().float()
        encoder_output, encoder_states = self.encoder(x_copy)
        # Reduce encoder hidden states.
        decoder_states = self.reduce_state(encoder_states)

        # Calculate loss for every step.
        step_losses = []
        for t in range(y.shape[1] - 1):
            # Do teacher forcing.
            x_t = y[:, t]
            x_t = replace_oovs(x_t, self.v)
            y_t = y[:, t + 1]

            # Get context vector from the attention network.
            context_vector, attention_weights = self.attention(
                decoder_states, encoder_output, x_padding_masks
            )
            # Get vocab distribution and hidden states from the decoder.
            p_vocab, decoder_states = self.decoder(
                x_t.unsqueeze(1), decoder_states, context_vector
            )
            # Get the probabilities predict by the model for target tokens.
            y_t = replace_oovs(y_t, self.v)
            target_probs = torch.gather(p_vocab, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)
            # Apply a mask such that pad zeros do not affect the loss
            mask = mask = torch.ne(y_t, 0).byte()
            # Do smoothing to prevent getting NaN loss because of log(0).
            loss = -torch.log(target_probs + config.eps)
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)
        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss

    def get_final_distribution(self, p_vocab):
        return p_vocab

    def load_model(self):
        """Load saved model if there exits one.
        """
        if (os.path.exists(config.encoder_save_name)):
            self.encoder = torch.load(config.encoder_save_name, map_location=torch.device('cpu'))
            self.decoder = torch.load(config.decoder_save_name, map_location=torch.device('cpu'))
            self.attention = torch.load(config.attention_save_name, map_location=torch.device('cpu'))
            self.reduce_state = torch.load(config.reduce_state_save_name, map_location=torch.device('cpu'))

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        """
        len_Y = len(self.tokens)
        # Lenth normalization
        ln = (5 + len_Y) ** config.alpha / (5 + 1) ** config.alpha
        cn = config.beta * torch.sum(  # Coverage normalization
            torch.log(
                config.eps +
                torch.where(
                    self.coverage_vector < 1.0,
                    self.coverage_vector,
                    torch.ones((1, self.coverage_vector.shape[1])).to(torch.device(config.DEVICE)))))

        score = sum(self.log_probs) / ln + cn

        # score = sum(self.log_probs) / ln
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()

import heapq
"""
heapq 实现了适用于 Python 列表的最小堆排序算法。

堆是一个树状的数据结构，其中的子节点与父节点属于排序关系。可以使用列表或数组来表示二进制堆，使得元素 N 的子元素位于 2 * N + 1 和 2 * N + 2 的位置（对于从零开始的索引）。这种布局使得可以在适当的位置重新排列堆，因此在添加或删除数据时无需重新分配内存。

max-heap 确保父级大于或等于其子级。min-heap 要求父项小于或等于其子级。Python 的heapq模块实现了一个 min-heap。
"""
def add2heap(heap, item, k):
    """Maintain a heap with k nodes and the smallest one as root.
    """
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)


class Predict():
    def __init__(self):
        DEVICE = config.DEVICE

        dataset = PairDataset(config.data_path)

        self.vocab = dataset.build_vocab(embed_file=config.embed_file)

        self.model = Seq2seq(self.vocab)
        self.stop_word = list(
            set([
                self.vocab[x.strip()] for x in
                open(config.stop_word_file, encoding='UTF-8'
                     ).readlines()
            ]))
        self.model.load_model()
        self.model.to(DEVICE)

    def greedy_search(self,
                      x,
                      max_sum_len,
                      len_oovs,
                      x_padding_masks):
        """Function which returns a summary by always picking the highest probability option conditioned on the previous word.
        """

        # Get encoder output and states.
        encoder_output, encoder_states = self.model.encoder(
            replace_oovs(x, self.vocab))

        # Initialize decoder's hidden states with encoder's hidden states.
        decoder_states = self.model.reduce_state(encoder_states)

        # Initialize decoder's input at time step 0 with the SOS token.
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(config.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS]
        # Generate hypothesis with maximum decode step.
        while int(x_t.item()) != (self.vocab.EOS) \
                and len(summary) < max_sum_len:
            context_vector, attention_weights = \
                self.model.attention(decoder_states,
                                     encoder_output,
                                     x_padding_masks)
            p_vocab, decoder_states = \
                self.model.decoder(x_t.unsqueeze(1),
                                   decoder_states,
                                   context_vector)
            final_dist = self.model.get_final_distribution(p_vocab)
            # Get next token with maximum probability.
            x_t = torch.argmax(final_dist, dim=1).to(config.DEVICE)
            decoder_word_idx = x_t.item()
            summary.append(decoder_word_idx)
            x_t = replace_oovs(x_t, self.vocab)

        return summary

    @timer('predict')
    def predict(self, text, tokenize=True):
        """Generate summary.
        """
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        x, oov = src2ids(text, self.vocab)
        x = torch.tensor(x).to(config.DEVICE)
        len_oovs = torch.tensor([len(oov)]).to(config.DEVICE)
        x_padding_masks = torch.ne(x, 0).byte().float()
        summary = self.greedy_search(x.unsqueeze(0),
                                     max_sum_len=config.max_dec_steps,
                                     len_oovs=len_oovs,
                                     x_padding_masks=x_padding_masks)
        summary = outputids2words(summary,
                                  oov,
                                  self.vocab)
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()

@__APP__.route("/predict", methods=["POST"])
def predict():
    rsp = {
        "success": True
    }
    try:
        data = json.loads(request.data)
        if "text" not in data.keys() or not isinstance(data["text"], str) or len(data["text"]) == 0:
            rsp["success"] = False
            rsp["info"] = "不合法的文本输入（text）！"
            return json.dumps(rsp, ensure_ascii=False, indent=2)
        global __PREDICTOR__
        if __PREDICTOR__ is None:
            rsp["success"] = False
            rsp["info"] = "服务尚未准备好，请稍等！"
        else:
            rsp["result"] = __PREDICTOR__.predict(data["text"].split())
    except RuntimeError as runtime_error:
        rsp["success"] = False
        rsp["info"] = "未知错误 {0}.".format(str(runtime_error))
    except json.decoder.JSONDecodeError as json_decode_error:
        rsp["success"] = False
        rsp["info"] = "解析输入错误 {0}.".format(str(json_decode_error))
    except Exception as exception:
        rsp["success"] = False
        rsp["info"] = str(exception)
    rsp = Response(json.dumps(rsp, ensure_ascii=False, indent=2))
    rsp.headers["content-type"] = "application/json"
    rsp.headers["Access-Control-Allow-Origin"] = "*"
    rsp.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept"

    return rsp

def serve(host: str="127.0.0.1", port: int=8080):
    global __PREDICTOR__, __APP__
    __PREDICTOR__ = Predict()
    server = HTTPServer(WSGIContainer(__APP__))
    print("->    :: Web Service ::    ")
    print("service starts at \x1b[1;32;40mhttp://{0}:{1}\x1b[0m".format(host, port))
    server.listen(port, host)
    IOLoop.instance().start()


if __name__ == "__main__":
    serve()
