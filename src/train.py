import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
from pathlib import Path
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras import optimizers
from .model import build_model, build_inference_model
from .sample import TextSampler, KeywordSampler, take, random_subsequence


def main():
    seq_len = 64
    batch_size = 64
    data_path = Path('preprocessed')
    output_path = output_path_name()

    raw_text = load_raw_text(data_path)
    train_batches, char2index = load_data(raw_text, seq_len, batch_size)
    model = build_model(batch_size=batch_size, embedding_dim=32,
                        num_lstm_layers=2, lstm_dim=192,
                        dropout_proportion=0.0, seq_len=seq_len,
                        vocab_size=len(char2index))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(clipnorm=5.0),
                  metrics=['accuracy'])

    output_path.mkdir()
    print(f'Saving the model to {output_path}')
    save_net(model, char2index, output_path)

    cb = callbacks(output_path, raw_text, model, char2index)
    hist = model.fit_generator(train_batches, epochs=400, shuffle=False,
                               callbacks=cb, verbose=1)

    save_weights(model, output_path)
    save_training_history(hist, output_path)


def callbacks(output_path, text, model, char2idx):
    checkpoint_template = str(output_path / 'weights.{epoch:02d}.hdf5')
    return [
        ResetStatesCallback(),
        ModelCheckpoint(checkpoint_template),
        PrintSampleCallback(text, model, char2idx)
    ]


def output_path_name():
    return (Path('weights') /
            datetime.datetime.now().replace(microsecond=0).isoformat())


def load_raw_text(data_path):
    return '\n'.join([
        p.open(encoding='utf-8-sig').read()
        for p in data_path.iterdir()
    ])


def load_data(raw_text, sequence_len, batch_size):
    unique_chars = frozenset(raw_text)
    char2index = dict((c, i) for i, c in enumerate(unique_chars))

    sequencesX = []
    sequencesY = []
    for X, Y in split_to_sequences(raw_text, sequence_len,
                                   sequence_len, char2index):
        sequencesX.append(X)
        sequencesY.append(Y)

    batches = OneHotLabelBatchSequence(sequencesX, sequencesY,
                                       len(unique_chars), batch_size)

    print('Total characters: {}'.format(len(raw_text)))
    print('Unique characters: {}'.format(len(unique_chars)))
    print('Number of train sequences: {}'.format(len(sequencesX)))
    print('Batch size: {}'.format(batch_size))
    print('Number of batches: {}'.format(len(batches)))

    return batches, char2index


def split_to_sequences(text, sequence_len, step, char2index):
    for i in range(0, len(text) - sequence_len, step):
        seq_in = [char2index[c] for c in text[i:(i + sequence_len)]]
        seq_out = [char2index[c] for c in text[(i + 1):(i + sequence_len + 1)]]
        yield (seq_in, seq_out)


def save_net(model, char2index, output_path):
    with (output_path / 'model.json').open('w') as f:
        f.write(model.to_json())

    with (output_path / 'char2idx.json').open('w') as f:
        json.dump(char2index, f)


def save_weights(model, output_path):
    model.save_weights(str(output_path / 'weights.hdf5'))

    tfjs_output_path = output_path / 'tfjs'
    tfjs_output_path.mkdir(exist_ok=True)
    tfjs.converters.save_keras_model(model, tfjs_output_path)


def save_training_history(hist, output_path):
    if 'loss' in hist.history:
        plt.plot(hist.history['loss'], label='train')
    if 'val_loss' in hist.history:
        plt.plot(hist.history['val_loss'], label='test')
    plt.title('Loss')
    plt.legend()
    plt.savefig(output_path / 'loss.png')
    plt.close()

    if 'acc' in hist.history:
        plt.plot(hist.history['acc'], label='train')
    if 'val_acc' in hist.history:
        plt.plot(hist.history['val_acc'], label='test')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(output_path / 'accuracy.png')
    plt.close()


class OneHotLabelBatchSequence(Sequence):
    def __init__(self, x, y, num_classes, batch_len):
        super().__init__()

        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.batch_len = batch_len

    def __len__(self):
        # Ignore the last partial batch
        return len(self.x) // self.batch_len

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_len:(idx + 1) * self.batch_len]
        batch_y = self.y[idx * self.batch_len:(idx + 1) * self.batch_len]
        return (np.array(batch_x),
                to_categorical(batch_y, num_classes=self.num_classes))


class PrintSampleCallback(Callback):
    def __init__(self, text, model, char2idx):
        super().__init__()
        self.text = text
        self.inference_model = build_inference_model(model)
        self.sampler = TextSampler(self.inference_model, char2idx)

    def on_epoch_end(self, epoch, logs=None):
        self._print_sampled(300)

    def on_train_end(self, logs=None):
        self._print_sampled(1000)

    def _print_sampled(self, n):
        seed = random_subsequence(self.text)
        print()
        print('generating with seed: {}'.format(seed))
        print('-'*40)

        self.inference_model.set_weights(self.model.get_weights())
        self.sampler.reset_states()
        keywords = KeywordSampler([])
        characters = self.sampler.character_generator(1.0, seed, keywords)
        print(seed + ''.join(take(n, characters)))


class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()


if __name__ == '__main__':
    main()
