from data_loader import load_data
from matplotlib import pyplot as plt
from utils.image import make_synthesized_image_pair
import tensorflow as tf

def main():
    print(tf.executing_eagerly())
    config = {}
    config['data_dir'] = '/home/dan/prj/datasets/mini_imagenet'
    config['dataset_name'] = 'mini_imagenet'

    ds = load_data(['val'], config)
    val_ds = ds['val'].shuffle(100)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for image1,image2, moving_vector in val_ds.take(1):
      ax1.imshow(image1)
      ax2.imshow(image2)
      print(moving_vector)
    plt.show()
if __name__ == '__main__':
    main()
