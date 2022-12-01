def encoder_stack(EncoderCtor, num_encoders):
    if (num_encoders == 0):
        return None
    return EncoderCtor(encoder_stack(EncoderCtor, num_encoders - 1))


class EncoderCtor:
    def __init__(self, next_encoder):
        self.next = next_encoder


start_encoder = encoder_stack(EncoderCtor, 6)
print(start_encoder)
