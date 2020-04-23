import numpy as np
from data_loader import load_data
from utils import image
from . import visualize
def synthesize_image_pair(config, splits):
    datasets = load_data(splits, config)
    ds = datasets['train'].batch(5)
    for image_A, image_B, parameters in ds.take(1):
        image_A = image_A.numpy()
        image_B = image_B.numpy()
        parameters = parameters.numpy()

    image_C = list(map(lambda x : image.synthesize_image(x[0], x[1], (64, 64), bbox=None, pad_ratio=None),
                   zip(image_A.copy(), parameters.copy())))
    image_C = np.array(image_C)   
    visualize.show_image([image_A, image_B, image_C])