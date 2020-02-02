import argparse
import json
import os.path
import numpy as np
from typing import Iterator, List, Optional, Tuple
from itertools import takewhile, islice
from keras.models import model_from_json
from .model import build_inference_model


class KeywordSampler:
    def __init__(self, keywords: List[str]):
        self.keywords = keywords
        self.weights = np.array([0.5]*len(keywords))

    def sample(self) -> Optional[str]:
        if not self.keywords or np.random.rand() > self.weights.max():
            return None

        p = self.weights/self.weights.sum()
        i = np.random.choice(range(len(self.keywords)), p=p)
        self.weights[i] = max(0.2*self.weights[i], 0.01)
        return self.keywords[i]


class TextSampler:
    def __init__(self, inference_model, char2idx):
        self.model = inference_model
        self.char2idx = char2idx
        self.idx2char = reverse_char2idx(char2idx)

    def sample_verses(self,
                      temperature: float,
                      min_length: int,
                      seed:str=None,
                      keywords: KeywordSampler=None) -> str:
        """Sample complete verses from the model."""

        keywords = keywords or KeywordSampler([])
        characters = self.character_generator(temperature, seed, keywords)
        if not seed:
            # Skip the first verse (line) as an initialization.
            # list() is necessary to force an eager evaluation.
            list(takewhile(not_line_feed, characters))

        body = ''.join(take(min_length, characters))

        # Continue sampling until the next line feed
        last_completion = ''.join(takewhile(not_line_feed, characters))

        return (body + last_completion).strip()

    def character_generator(self,
                            temperature: float,
                            seed: str,
                            keywords: KeywordSampler) -> Iterator[str]:
        """Generator that draws characters from the model forever."""
        if not seed:
            seed = random_character()

        # initialize the internal states
        self.advance(seed[:-1])

        line_break = False
        i = self.encode_text(seed[-1])[0]
        while True:
            if line_break:
                kw = keywords.sample()
                if kw:
                    line_break = False
                    self.advance(kw)
                    i = self.encode_text(kw[-1])[0]
                    yield from kw
                    continue

            c, i = self.sample_next_letter(i, temperature)
            line_break = c == '\n'

            yield c

    def advance(self, text: str) -> None:
        for i in self.encode_text(text):
            self.model.predict(np.array([[i]]))

    def reset_states(self) -> None:
        self.model.reset_states()

    def sample_next_letter(self, current_index: int, temperature: float) -> Tuple[str, int]:
        preds = self.model.predict(np.array([[current_index]]))
        p = np.exp(np.log(np.maximum(preds[0, -1, :], 1e-40)) / temperature)
        if np.sum(p) < 1e-16:
            # numerical stability
            p = np.zeros(len(p))
            p[np.argmax(preds[0, -1, :])] = 1.0
        else:
            p = p / np.sum(p)
        i = np.random.choice(range(len(self.idx2char)), p=p)
        return self.idx2char[i], i

    def encode_text(self, text: str):
        return np.fromiter((self.char2idx.get(c, 0) for c in text), int)


def main():
    """Sample text from a pre-trained model."""
    args = parse_args()
    sampler = load_sampler(args.model_path)
    keywords = KeywordSampler([])
    characters = sampler.character_generator(args.temperature, args.preseed, keywords)
    text = ''.join(take(args.n, characters))
    text = args.preseed + text
    print(text)


def load_sampler(model_path):
    model, char2idx = load_inference_model(model_path)
    return TextSampler(model, char2idx)


def load_inference_model(path):
    model_json = open(os.path.join(path, 'model.json')).read()
    model = model_from_json(model_json)
    model = build_inference_model(model)
    model.load_weights(os.path.join(path, 'weights.hdf5'))
    char2idx = json.load(open(os.path.join(path, 'char2idx.json')))
    return model, char2idx


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
