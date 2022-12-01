import torch
import torch.nn.functional as F
import torch.nn as nn


def coder_stack(CoderCtor, num_encoders):
    if (num_encoders == 0):
        return None
    return CoderCtor(coder_stack(CoderCtor, num_encoders - 1))


class EncoderCtor(nn.Module):
    def __init__(self, next):
        super(EncoderCtor, self).__init__()
        self.next = next
        self_attention_layer = nn.Linear(1, 1, bias=True)
        feedforward_layer = nn.Linear(1, 1, bias=True)
        self.stack = nn.Sequential(self_attention_layer, feedforward_layer)


class DecoderCtor(nn.Module):
    def __init__(self, next):
        super(DecoderCtor, self).__init__()
        self_attention_layer = nn.Linear(1, 1, bias=True)
        encoder_decoder_attention_layer = nn.Linear(1, 1, bias=True)
        feedforward_layer = nn.Linear(1, 1, bias=True)
        self.stack = nn.Sequential(self_attention_layer, encoder_decoder_attention_layer, feedforward_layer)


word_width = 512
W_Q = torch.randn([word_width, 64]) / 100
W_K = torch.randn([word_width, 64]) / 100
W_V = torch.randn([word_width, 64]) / 100


def qkv(word):
    return torch.matmul(word, W_Q), torch.matmul(word, W_K), torch.matmul(word, W_V)

def qkvs(words):
    return torch.matmul(words, W_Q), torch.matmul(words, W_K), torch.matmul(words, W_V)


def attention_score(qkv):
    return torch.dot(qkv[0], qkv[1]) / 8

softmax = torch.nn.Softmax(dim=1)
def attention_scores(qkvs):
    return torch.matmul(softmax(torch.matmul(qkvs[0], torch.transpose(qkvs[1], 0, 1)) / 8.), qkvs[2])


def softmax_scores(scores):
    softmax = torch.nn.Softmax(dim=0)
    return softmax(scores)


start_encoder = coder_stack(EncoderCtor, 6)
start_decoder = coder_stack(DecoderCtor, 6)

num_words = 2
word = torch.ones(word_width)
words = torch.randn([num_words, word_width])
qkv_words = qkvs(words)
print(attention_scores(qkv_words))

