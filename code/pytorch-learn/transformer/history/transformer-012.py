import torch
import torch.nn as nn

softmax = torch.nn.Softmax(dim=1)
num_heads = 8
word_width = 512
projection_width = 64


def coder_stack(CoderCtor, num_encoders):
    if (num_encoders == 0):
        return None
    return CoderCtor(coder_stack(CoderCtor, num_encoders - 1))


class SelfAttentionLayer:
    def __init__(self, w_q, w_k, w_v):
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v

    def forward(self, words):
        return attention_scores(qkvs(words, self.w_q, self.w_k, self.w_v))


class EncoderCtor(nn.Module):
    def __init__(self, w_o):
        super(EncoderCtor, self).__init__()
        self.w_o = w_o
        self.attention_layers = list(map(lambda x: SelfAttentionLayer(W_Q, W_K, W_V), range(num_heads)))
        self.feedforward_layer = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        # Concatenating gives [num_words x num_heads * projection_width]
        attention_vectors = list(map(lambda attention_layer: attention_layer.forward(x), self.attention_layers))
        return torch.matmul(torch.cat(attention_vectors, dim=1), self.w_o)


class DecoderCtor(nn.Module):
    def __init__(self, next):
        super(DecoderCtor, self).__init__()
        self_attention_layer = nn.Linear(1, 1, bias=True)
        encoder_decoder_attention_layer = nn.Linear(1, 1, bias=True)
        feedforward_layer = nn.Linear(1, 1, bias=True)
        self.stack = nn.Sequential(self_attention_layer, encoder_decoder_attention_layer, feedforward_layer)


W_Q = torch.randn([word_width, projection_width]) / 100
W_K = torch.randn([word_width, projection_width]) / 100
W_V = torch.randn([word_width, projection_width]) / 100
W_O = torch.randn([num_heads * projection_width, word_width]) / 100

def qkvs(words, w_q, w_k, w_v):
    return torch.matmul(words, w_q), torch.matmul(words, w_k), torch.matmul(words, w_v)


def attention_scores(qkvs):
    return torch.matmul(softmax(torch.matmul(qkvs[0], torch.transpose(qkvs[1], 0, 1)) / 8.), qkvs[2])


start_encoder = coder_stack(EncoderCtor, 6)
start_decoder = coder_stack(DecoderCtor, 6)

num_words = 2
word = torch.ones(word_width)
words = torch.randn([num_words, word_width])
qkv_words = qkvs(words, W_Q, W_K, W_V)
# print(attention_scores(qkv_words))
# print(SelfAttentionLayer(W_Q, W_K, W_V).forward(words))

encoder = EncoderCtor(W_O)
encoder.eval()
print(encoder(words).shape)
