import torch
import torch.nn.functional as F
import torch.nn as nn

def coder_stack(CoderCtor, num_encoders):
    if (num_encoders == 0):
        return None
    return CoderCtor(coder_stack(CoderCtor, num_encoders - 1))

class EncoderCtor:
    def __init__(self, next):
        self.next = next

class DecoderCtor:
    def __init__(self, next):
        self.next = next


start_encoder = coder_stack(EncoderCtor, 6)
start_decoder = coder_stack(DecoderCtor, 6)
print(start_encoder)
