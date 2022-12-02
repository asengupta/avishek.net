import torch
import torch.nn as nn

softmax = torch.nn.Softmax(dim=1)
num_heads = 8
word_width = 512
projection_width = 64
scale_factor = 100


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

class MultiheadedAttention(nn.Module):
    def __init__(self, num_heads, w_o, word_width):
        super(MultiheadedAttention, self).__init__()
        self.w_o = w_o
        self.attention_layers = list(map(lambda x: SelfAttentionLayer(W_Q, W_K, W_V), range(num_heads)))

    def forward(self, x):
        # Concatenating gives [num_words x num_heads * projection_width]
        attention_vectors = list(map(lambda attention_layer: attention_layer.forward(x), self.attention_layers))
        concatenated_attention_vectors = torch.cat(attention_vectors, dim=1)
        scaled_concatenated_attention_vectors = torch.matmul(concatenated_attention_vectors, self.w_o)
        return scaled_concatenated_attention_vectors

class EncoderCtor(nn.Module):
    def __init__(self, num_heads, w_o, word_width):
        super(EncoderCtor, self).__init__()
        self.w_o = w_o
        self.layer_norm = nn.LayerNorm(word_width)
        self.multiheaded_attention_layer = MultiheadedAttention(num_heads, w_o, word_width)
        self.feedforward_layer = nn.Sequential(nn.Linear(word_width, 2048, bias=True), nn.LeakyReLU(), nn.Linear(2048, word_width, bias=True))

    def forward(self, x):
        mh_output = self.multiheaded_attention_layer(x)
        layer_normed_multihead = self.layer_norm(mh_output + x)
        ffnn_outputs = torch.stack(list(map(lambda attention_vector: self.feedforward_layer(attention_vector), layer_normed_multihead)))
        layer_normed_ffnn = self.layer_norm(ffnn_outputs + layer_normed_multihead)
        return layer_normed_ffnn


class DecoderCtor(nn.Module):
    def __init__(self, next):
        super(DecoderCtor, self).__init__()
        self_attention_layer = nn.Linear(1, 1, bias=True)
        encoder_decoder_attention_layer = nn.Linear(1, 1, bias=True)
        feedforward_layer = nn.Linear(1, 1, bias=True)
        self.stack = nn.Sequential(self_attention_layer, encoder_decoder_attention_layer, feedforward_layer)


W_Q = torch.randn([word_width, projection_width]) / scale_factor
W_K = torch.randn([word_width, projection_width]) / scale_factor
W_V = torch.randn([word_width, projection_width]) / scale_factor
W_O = torch.randn([num_heads * projection_width, word_width]) / scale_factor

def qkvs(words, w_q, w_k, w_v):
    return torch.matmul(words, w_q), torch.matmul(words, w_k), torch.matmul(words, w_v)


def attention_scores(qkvs):
    return torch.matmul(softmax(torch.matmul(qkvs[0], torch.transpose(qkvs[1], 0, 1)) / 8.), qkvs[2])


# start_encoder = coder_stack(EncoderCtor, 6)
# start_decoder = coder_stack(DecoderCtor, 6)

num_words = 2
words = torch.randn([num_words, word_width])
qkv_words = qkvs(words, W_Q, W_K, W_V)
encoder = EncoderCtor(num_heads, W_O, word_width)
encoder.eval()
values = encoder(words)
print(values)
print(values.shape)
