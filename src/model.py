from keras.models import Sequential, load_model
from keras.layers import LSTM, CuDNNLSTM, Dropout, TimeDistributed, Dense, \
    Activation, Embedding


def build_model(batch_size, embedding_dim, num_lstm_layers, lstm_dim,
                dropout_proportion, seq_len, vocab_size, gpu=False):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim,
                        batch_input_shape=(batch_size, seq_len)))
    model.add(Dropout(dropout_proportion))
    for _ in range(num_lstm_layers):
        lstm = CuDNNLSTM if gpu else LSTM
        model.add(lstm(lstm_dim, return_sequences=True, stateful=True))
        model.add(Dropout(dropout_proportion))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    return model


def build_inference_model(model):
    config = model.get_config()
    config['layers'][0]['config']['batch_input_shape'] = (1, 1)
    inference_model = Sequential.from_config(config)
    inference_model.trainable = False
    return inference_model


if __name__ == '__main__':
    model = build_model(32, 128, 3, 256, 0.2, 64, 26)
    model.summary()
