from test import data, train
import json, os

def test_data():
    with open("overfit.json") as fp:
        config = json.load(fp)
    splits = ['train']
    data.synthesize_image_pair(config, splits)

def test_train():
    with open("overfit.json") as fp:
        config = json.load(fp)
    train.overfit(config, ['train'])
    #train.result_test(config, ['train'])
if __name__ == "__main__":
    test_train()