import chainer
from chainer import links as L
from chainer import functions as F
from chainer import optimizers, cuda
import numpy as np
import process_data
import pickle
import yaml
from _datetime import datetime
import os
from logging import getLogger, StreamHandler, INFO, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)


class RNN(chainer.Chain):
    def __init__(self, vocab_size, embed_size, lstm_layers, hidden_size, dropout, ARR):
        super(RNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, embed_size, ignore_label=-1)
            self.lstm = L.NStepLSTM(n_layers=lstm_layers, in_size=embed_size, out_size=hidden_size, dropout=dropout)
            self.l2 = L.Linear(hidden_size, 1)
        self.ARR = ARR
        self.embed_size = embed_size

    def __call__(self, xs):
        xs = [self.embed(x) for x in xs]
        hy, cy, _ = self.lstm(xs=xs, hx=None, cx=None)
        y = F.sigmoid(self.l2(hy[-1]))
        return y

    def predict(self, x):
        return self([x])[0]


def get_train_data():
    data = process_data.get_train_data()
    vocab = dict()
    current_id = 0
    for line in data:
        tweet = line[0]
        # 今回は文字単位でIDと対応付ける
        for c in tweet:
            if c in vocab:
                continue
            vocab[c] = current_id
            current_id += 1
    return [([vocab[c] for c in line[0]], line[1]) for line in data], vocab


def main():
    config = yaml.load(open("config.yml", encoding="utf-8"))
    if os.path.exists(config.get("VOCAB_FILE", "jk_vocab.pkl")) and os.path.exists(
            config.get("TRAIN_DATA_FILE", "train_data.pkl")):
        with open(config.get("VOCAB_FILE", "jk_vocab.pkl"), "rb") as f:
            voc = pickle.load(f)
        with open(config.get("TRAIN_DATA_FILE", "train_data.pkl"), "rb") as f:
            train_data = pickle.load(f)
    else:
        train_data, voc = get_train_data()
        with open(config.get("VOCAB_FILE", "jk_vocab.pkl"), "wb") as f:
            pickle.dump(voc, f)
        with open(config.get("TRAIN_DATA_FILE", "train_data.pkl"), "wb") as f:
            pickle.dump(train_data, f)
    logger.info("data length:{}, vocab length:{}".format(len(train_data), len(voc)))

    VOCAB_SIZE = len(voc)
    EMBED_SIZE = int(config.get("EMBED_SIZE", 500))
    HIDDEN_SIZE = int(config.get("HIDDEN_SIZE", 100))
    LSTM_LAYERS = int(config.get("LSTM_LAYERS", 3))
    DROPOUT = float(config.get("DROPOUT", 0.5))
    EPOCH_NUM = int(config.get("EPOCH_NUM", 30))
    BATCH_SIZE = int(config.get("BATCH_SIZE", 20))
    MODEL_FILE_NAME = config.get("MODEL_FILE", "jk_model.npz")
    GPU = config.get("GPU", False)
    xp = cuda.cupy if GPU else np

    model = RNN(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, lstm_layers=LSTM_LAYERS, hidden_size=HIDDEN_SIZE,
                dropout=DROPOUT, ARR=xp)

    logger.info("========== START ==========")
    logger.info(datetime.now())
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    for i in range(EPOCH_NUM):
        if GPU:
            gpu_device = 0
            cuda.get_device(gpu_device).use()
            model.to_gpu(gpu_device)
        logger.info("========== EPOCH {} ==========".format(i + 1))
        total_loss = 0
        for j in range(0, len(train_data), BATCH_SIZE):
            x = [xp.asarray(data[0], dtype=xp.int32) for data in train_data[j:j + BATCH_SIZE]]
            t = [xp.asarray(data[1], dtype=xp.float32) for data in train_data[j:j + BATCH_SIZE]]

            logger.debug("x: {}".format(x))
            logger.debug("t: {}".format(t))
            y = model(x)
            loss = None
            for yi, ti in zip(y, t):
                logger.debug("predict:{} , answer:{}".format(yi, ti))
                if loss is not None:
                    loss += F.mean_squared_error(yi, ti)
                else:
                    loss = F.mean_squared_error(yi, ti)
            total_loss += loss.data
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            logger.info("loss: {}  - \t {} / {}".format(loss.data / len(x), j, len(train_data)))
        logger.info("total_loss:{}".format(total_loss / len(train_data)))
        model.to_cpu()
        chainer.serializers.save_npz(MODEL_FILE_NAME, model)

    logger.info("========== END ==========")
    logger.info(datetime.now())


if __name__ == '__main__':
    main()
