from test import data, train
import json, os

def test_data(config):
    with open("overfit.json") as fp:
        config = json.load(fp)
    splits = ['train']
    data.synthesize_image_pair(config, splits)

def test_train(config):
    with open("overfit.json") as fp:
        config = json.load(fp)
    #train.overfit(config, ['train'])
    train.result_test(config, ['train'])
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    config = args.config
    test_train(config)
    #test_data(config)