import argparse
import json
import os.path
import numpy as np
from itertools import takewhile, islice
from keras.models import model_from_json
from .model import build_inference_model


class TextSampler():
    def __init__(self, path):
        self.model, self.char2idx, self.idx2char = load_inference_model(path)

    def character_generator(self, temperature, seed=None):
        """Generator that draws characters from the model forever."""
        self.model.reset_states()

        if not seed:
            seed = random_character()
        encoded = encode_text(seed, self.char2idx)

        # initialize the internal states
        for i in encoded[:-1]:
            x = np.array([[i]])
            self.model.predict(x)

        next_index = encoded[-1]
        while True:
            x = np.array([[next_index]])
            preds = self.model.predict(x)
            p = np.exp(np.log(np.maximum(preds[0, -1, :], 1e-40)) / temperature)
            p = p / np.sum(p)
            next_index = np.random.choice(range(len(self.idx2char)), p=p)

            yield self.idx2char[next_index]

    def sample_verses(self, temperature, min_length, seed=None):
        characters = self.character_generator(temperature, seed)
        if not seed:
            # Skip the first verse (line) because it often incomplete.
            # list() is necessary to force the evaluation.
            list(takewhile(not_line_feed, characters))

        body = ''.join(take(min_length, characters))

        # Complete the last verse
        last_completion = ''.join(takewhile(not_line_feed, characters))

        return (body + last_completion).strip()


def main():
    """Sample text from a pre-trained model."""
    args = parse_args()
    sampler = TextSampler(args.model_path)
    characters = sampler.character_generator(args.temperature, args.preseed)
    text = ''.join(take(args.n, characters))
    text = args.preseed + text

    print(text)


def encode_text(text, char2idx):
    return np.fromiter((char2idx.get(c, 0) for c in text), int)


def load_inference_model(path):
    model_json = open(os.path.join(path, 'model.json')).read()
    model = model_from_json(model_json)
    model = build_inference_model(model)
    model.load_weights(os.path.join(path, 'weights.hdf5'))
    char2idx = json.load(open(os.path.join(path, 'char2idx.json')))
    idx2char = reverse_char2idx(char2idx)
    return model, char2idx, idx2char


def reverse_char2idx(char2idx):
    idx2char = [None for i in range(len(char2idx))]
    for (c, i) in char2idx.items():
        idx2char[i] = c
    return idx2char


def random_subsequence(text, seq_lens=(8, 16, 32)):
    seq_len = np.random.choice(seq_lens)
    start = np.random.randint(0, len(text) - seq_len - 1)
    return text[start:(start + seq_len)]


def random_character():
    return np.random.choice([chr(x) for x in range(ord('A'), ord('Z'))])


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def not_line_feed(c):
    return c != '\n'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',
                        help='The location of the learned model')
    parser.add_argument('-n', default=500, type=int,
                        help='Number of characters to sample')
    parser.add_argument('-t', '--temperature', type=float, default=1.0,
                        help='The sampling temperature. Smaller values mean '
                        'more conservative predictions')
    parser.add_argument('--preseed',
                        help='Use this text as a seed for the sampling')
    return parser.parse_args()


if __name__ == '__main__':
    main()
