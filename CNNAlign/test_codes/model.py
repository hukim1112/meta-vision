import tensorflow as tf
import numpy as np
from data_loader import load_data
from utils import image
from models.cnn_geo import CNN_geo
from . import visualize
def output(config, splits):
    model = CNN_geo("prototypical_network")
    # ckpt_dir = os.path.join(
    #     'checkpoints', config['model_name'], config['exp_desc'])
    # ckpt = os.path.join(ckpt_dir, "{}-{}.h5".format(config['model_name'], str(config['train']['epochs'])))
    # model.load(ckpt)
    datasets = load_data(splits, config)
    ds = datasets['train'].batch(5)
    for image_A, image_B, parameters in ds.take(1):
        image_A = image_A.numpy()
        image_B = image_B.numpy()
        parameters = parameters.numpy()
    pred, _ = model(image_A, image_B)
    print(pred, pred.shape)

    pseudo_labels = (tf.random.uniform([5, 9, 2]) - 0.5) * 2 *0.2

    loss = tf.reduce_sum(tf.pow(pred-pseudo_labels, 2), axis=[-1,-2])

    _in = tf.ones([5, 9, 2])*2
    print(_in)
    _out = tf.ones([5, 9, 2])
    loss = tf.sqrt(tf.reduce_sum(tf.pow(_in-_out, 2), axis=[-1,-2]))
    #loss = tf.keras.losses.MSE(pred, pseudo_labels)
    print(loss)


    # print("compare gt : {} and pred : {}".format(parameters, pred))
    # loss = tf.reduce_sum(tf.keras.losses.MSE(pred, parameters), axis=1)
    # print("loss : {}".format(loss))

    # pred = pred.numpy()
    # image_C = list(map(lambda x : image.synthesize_image(x[0], x[1], (64, 64), bbox=None, pad_ratio=None),
    #                zip(image_A.copy(), pred.copy())))    
    # image_C = np.array(image_C)
    # visualize.show_image([image_A, image_B, image_C])    