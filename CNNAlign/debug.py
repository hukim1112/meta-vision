from test import data, train
import json, os
import argparse

def test_data(config):
    splits = ['train']
    data.synthesize_image_pair(config, splits)

def test_train(config):
    train.overfit(config, ['train'])
    #train.result_test(config, ['train'])
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    config = args.config
    with open(config) as fp:
        config = json.load(fp)
    print(config)
    test_train(config)
    #test_data(config)