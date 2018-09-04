from pymongo import MongoClient
import re
import random
import sys

sys.path.append('../../')
from util import tweet_preprocess  # Twitter関連のモジュールだったりする

DB_HOST = "localhost"
DB_PORT = 27017
DB_NAME = "twitter"
DB_COLLECTION_NAME = "tweet"


def get_data():
    """
    MongoDBに接続してJKのツイートとそうでないツイートを同数抽出
    :return:
    """
    with MongoClient(DB_HOST, DB_PORT) as client:
        db = client[DB_NAME]
        collection = db[DB_COLLECTION_NAME]
    jk_tweet_data = []
    normal_tweet_data = []
    for cursor in collection.find():
        tweet = cursor.get("tweet")
        user = cursor.get("user")
        if not tweet or not user:
            continue
        # ツイートからユーザID・URL・ハッシュタグ・改行を除いておきます
        tweet = tweet_preprocess.normalize(tweet)
        if len(tweet) < 2:
            continue
        description = user.get("description")
        # "JK[0-9]+"表記がプロフィールに含まれるユーザのツイートを取得
        if description and re.match(".*JK[0-9]+.*", description):
            jk_tweet_data.append(tweet)
        elif len(normal_tweet_data) < len(jk_tweet_data) and random.randint(1, 10) == 1:
            normal_tweet_data.append(tweet)

    return jk_tweet_data, normal_tweet_data


def get_train_data():
    jk_train_data, normal_train_data = get_data()
    jk_train_data = [(tweet, [1]) for tweet in jk_train_data]
    normal_train_data = [(tweet, [0]) for tweet in normal_train_data]
    train_data = list(jk_train_data + normal_train_data)
    random.shuffle(train_data)
    return train_data
