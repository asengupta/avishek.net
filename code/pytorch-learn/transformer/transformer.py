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

W_Q = torch.ones([512, 64])
W_K = torch.ones([512, 64])
W_V = torch.ones([512, 64])
def qkv(input):
    return (torch.matmul(input, W_Q), torch.matmul(input, W_K), torch.matmul(input, W_V))

def attention_score(qkv):
    return torch.dot(qkv[0], qkv[1]) / 8

def softmax_scores(scores):
    softmax = torch.nn.Softmax(dim=0)
    return softmax(scores)

start_encoder = coder_stack(EncoderCtor, 6)
start_decoder = coder_stack(DecoderCtor, 6)

tensors = torch.tensor([attention_score(qkv(torch.ones(512))),
                        attention_score(qkv(torch.ones(512))),
                        attention_score(qkv(torch.ones(512))), ])
print(softmax_scores(tensors))
