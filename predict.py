import chainer
import yaml
import pickle
import numpy as np
import train

config = yaml.load(open("config.yml", encoding="utf-8"))
import sys

sys.path.append('../../')
from util import tweet_preprocess  # Twitter関連のモジュールだったりする


def load_model():
    # モデル読み込み
    with open(config.get("VOCAB_FILE", "jk_vocab.pkl"), "rb") as f:
        voc = pickle.load(f)
    VOCAB_SIZE = len(voc)
    EMBED_SIZE = int(config.get("EMBED_SIZE", 200))
    HIDDEN_SIZE = int(config.get("HIDDEN_SIZE", 100))
    LSTM_LAYERS = int(config.get("LSTM_LAYERS", 3))
    DROPOUT = float(config.get("DROPOUT", 0.2))
    model_ = train.RNN(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, lstm_layers=LSTM_LAYERS,
                       dropout=DROPOUT, ARR=np)
    chainer.serializers.load_npz(config.get("MODEL_FILE", "jk_model.npz"), model_)
    return model_, voc


def predict(model_, vocab_, tweet, normalize=True):
    np.random.seed(765)
    if normalize:
        tweet = tweet_preprocess.normalize(tweet)
    tweet = np.asarray([vocab_.get(c, -1) for c in tweet], dtype=np.int32)
    y = model_.predict(tweet).data[0]
    return round(y * 100, 1)


if __name__ == '__main__':
    model, vocab = load_model()
    while True:
        print("JK RATE -> {} %".format(predict(model, vocab, input("INPUT>>"))))
