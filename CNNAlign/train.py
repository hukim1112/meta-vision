import os, json
from utils import parser, session_config
from pathlib import Path
from data_loader import load_data
from models import load_model
from matplotlib import pyplot as plt
import tensorflow as tf

def main():
    print(tf.executing_eagerly())

    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    ds = load_data(['train', 'val'], config)
    train_ds = ds['train'].batch(64)

if __name__ == '__main__':
    main()
