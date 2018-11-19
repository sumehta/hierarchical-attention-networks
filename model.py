import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class GRUEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, bidirectional=False, embedding=None):
        super(GRUEncoder, self).__init__()
        self.embedding = embedding
        self.gru = nn.GRU(emb_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)

    def forward(self, inp, init_states=None):
        batch_size = inp.size(0)  #bsz x num_sents
        emb_sequence = (self.embedding(inp) if self.embedding is not None
                            else inp)
        device = inp.device
        init_states = self.init_lstm_states(batch_size, device)

        gru_out, final_state = self.gru(emb_sequence, init_states)
        return gru_out

    def init_lstm_states(self, batch_size, device):
        return torch.zeros((2 if self.gru.bidirectional else 1), batch_size, self.gru.hidden_size).to(device)

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self.embedding.weight.size() == embedding.size()
        self.embedding.weight.data.copy_(embedding.data)

class SentClassifier(nn.Module):
    """
    Given a sentence concatenated with its document representation predicts
    whether it is protest related or not
    """
    def __init__(self, emb_size, nhid, dropout, nclasses):
        super(SentClassifier, self).__init__()
        self.hidden = nn.Linear(emb_size, nhid)
        self.hidden2op = nn.Linear(nhid, nclasses, bias=True)
        self.nclasses = nclasses
        self.dropout = nn.Dropout(dropout)
        # self.init_params()

    def forward(self, inp): #inp = [bsz, sent_emb+doc_emb]
        hid = self.dropout(self.hidden(inp))
        return self.hidden2op(hid)

class IntraAttention(nn.Module):
    def __init__(self, hidden_dim, bidirectional=True):
        super(IntraAttention, self).__init__()
        self.weight_W_word = nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), hidden_dim * (2 if bidirectional else 1)))
        self.bias_word = nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), 1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(hidden_dim * (2 if bidirectional else 1), 1))
        self.softmax_word = nn.Softmax(dim=1)
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.bias_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)

    def forward(self, inp):
        # dim of inp = 224 x 50 x hid_dim  repeat word weight for batchsize and perform parallel computation
        word_squish = torch.tanh(torch.bmm(self.weight_W_word.repeat(inp.size(0), 1, 1), inp.transpose(2,1)) + self.bias_word.repeat(inp.size(0), 1, inp.size(1)))
        # word_squish = bszx hid_dim x num_words
        word_attn_logits = torch.bmm(word_squish.transpose(2,1), self.weight_proj_word.repeat(inp.size(0), 1, 1)) #repeat the projection word for bsz times
        word_attn = self.softmax_word(word_attn_logits)
        # mask attn weights
        return word_attn.squeeze(2)

class HAN(nn.Module):
    def __init__(self, args):
        super(HAN, self).__init__()
        self.args = args
        embedding = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)
        self._sent_encoder_gru = GRUEncoder(args.emb_dim, args.lstm_hidden, args.bidirectional, embedding)
        self._doc_encoder_gru = GRUEncoder(args.lstm_hidden*(2 if args.bidirectional else 1), args.lstm_hidden, args.bidirectional)

        self.word_attn = IntraAttention(args.lstm_hidden, args.bidirectional)
        self.sent_attn = IntraAttention(args.lstm_hidden, args.bidirectional)

        self._sent_classifier = SentClassifier(args.lstm_hidden*(2 if args.bidirectional else 1), args.mlp_nhid, args.dropout, args.nclass) #300+500
        self._softmax = torch.nn.Softmax()

    def forward(self, inp, lens):
        in_ = inp.view(-1, inp.size()[-1])
        gru_out_word = self._sent_encoder_gru(in_)
        word_mask = ~(inp.view(inp.size(0)*inp.size(1), -1)==0)  # mask out 0 padded words
        word_attn_weights = self.word_attn(gru_out_word)
        word_attn_weights = word_attn_weights * word_mask.float()

        sent_emb = torch.sum(torch.mul(word_attn_weights.unsqueeze(2).repeat(1, 1, gru_out_word.size()[-1]).transpose(2,1), gru_out_word.transpose(2,1)), dim=2)  #224 x 500
        sent_emb = sent_emb.view(inp.size(0), inp.size(1), -1)

        gru_out_sent = self._doc_encoder_gru(sent_emb)
        sent_attn_weights = self.sent_attn(gru_out_sent)

        sent_mask = ~(torch.sum(~ (inp == 0), dim=2)==0)
        sent_attn_weights = sent_attn_weights * sent_mask.float()

        # weighted average of sent embs
        doc_emb = torch.sum(torch.mul(sent_attn_weights.unsqueeze(2).repeat(1, 1, gru_out_sent.size()[-1]).transpose(2,1), gru_out_sent.transpose(2,1)), dim=2)  #224 x 5
        logits = self._sent_classifier(doc_emb)

        return logits, word_attn_weights, sent_attn_weights #sigmoid is for binary

    def set_embedding(self, embedding):
        self._sent_encoder_gru.set_embedding(embedding)
