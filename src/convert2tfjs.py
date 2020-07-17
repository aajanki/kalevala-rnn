# Convert a trained Keras model into tensorflow.js model
#
# This is a stand-alone script to force Tensorflow to use same
# variables names as in weights.hdf5.

import json
import shutil
import plac
import tensorflowjs as tfjs
from pathlib import Path
from tensorflow.keras.models import model_from_json
from .model import build_inference_model


@plac.pos('model_path', 'Path to the trained model: weights/some/path/here')
def main(model_path):
    path = Path(model_path)
    model_json = json.load((path / 'model.json').open())

    # Set batch size to one
    model_json['config']['layers'][0]['config']['batch_input_shape'] = [1, 1]
    model_json['config']['build_input_shape'] = [1, 1]

    # Create the model from config and populate the weights
    model = model_from_json(json.dumps(model_json))
    model.load_weights(str(path / 'weights.hdf5'))

    # Save in tensorflow.js format
    output_path = path / 'tfjs'
    output_path.mkdir(exist_ok=True)
    tfjs.converters.save_keras_model(model, str(output_path))

    # copy characters
    shutil.copy(path / 'char2idx.json', output_path / 'char2idx.json')

    # The web app build process reads the model from the prod-model directory
    model_path = Path('prod-model')
    for child in output_path.iterdir():
        shutil.copy(child, model_path)


if __name__ == '__main__':
    plac.call(main)
