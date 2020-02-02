# Recurrent neural network text generation

This project trains a recurrent neural network (RNN) that generates
text imitating the style of
[Kalevala](https://en.wikipedia.org/wiki/Kalevala) and
[Kanteletar](https://en.wikipedia.org/wiki/Kanteletar).

A character level RNN is implemented using Keras. The trained model is
a stateful RNN with two LSTM layers.

## Usage

Prepare the environment:

```
pipenv install
pipenv shell
```

Training:

```
python src/train.py
```

Generating text from a trained model:

```
python -m src.sample -n 1000 --temperature 0.4 --preseed "Sanoi vanha Väinämöinen" weights/<path>
```

## HTTP server

Copy (and edit if necessary) the sample dotenv file to `.env`.
```
cp sample.env .env
```

Start the server:
```
uvicorn src.server:app --reload --env-file .env
```

## License

MIT license

The texts of Kalevala and Kanteletar included in the data subdirectory
are in public domain.
