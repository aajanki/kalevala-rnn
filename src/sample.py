import argparse
import itertools
import json
import os.path
import numpy as np
from keras.models import model_from_json
from model import build_inference_model


def main():
    """Sample text from a pre-trained model."""
    args = parse_args()
    model, char2idx, idx2char = load_inference_model(args.model_path)
    seed = args.preseed if args.preseed else random_character()
    characters = character_generator(
        model, args.temperature, idx2char, char2idx, seed)
    text = ''.join(take(args.n, characters))
    text = seed + text

    print(text)


def character_generator(model, temperature, idx2char, char2idx, seed):
    """Generator that draws characters from the model forever."""
    model.reset_states()
    encoded = encode_text(seed, char2idx)

    # initialize the internal states
    for i in encoded[:-1]:
        x = np.array([[i]])
        model.predict(x)

    next_index = encoded[-1]
    while True:
        x = np.array([[next_index]])
        preds = model.predict(x)
        p = np.exp(np.log(np.maximum(preds[0, -1, :], 1e-40)) / temperature)
        p = p / np.sum(p)
        next_index = np.random.choice(range(len(idx2char)), p=p)

        yield idx2char[next_index]


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
    return list(itertools.islice(iterable, n))


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
