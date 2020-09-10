import tensorflow as tf
from models.module import generate_inlier_mask

class CNN_semanticalign(tf.keras.Model):
    def __init__(self, cnngeo, geo_transform):
        super(CNN_semanticalign, self).__init__()
        self.model_name = 'CNNalign'
        self.cnngeo = cnngeo
        self.geo_transform = geo_transform
    def call(self, imageA, imageB):
        geo_parameters, correlations = self.cnngeo(imageA, imageB)
        map_size = correlations.shape[-2:] #H,W
        inlier_masks = generate_inlier_mask(geo_parameters, self.geo_transform, map_size)
        inlier_matching = correlations * inlier_masks  # B, H, W, H, W
        inlier_count = tf.reduce_sum(inlier_matching, axis=(1, 2, 3, 4))
        return correlations, inlier_matching, inlier_count
    def save(self, ckpt_path):
        self.cnngeo.save_weights(ckpt_path)
    def load(self, ckpt_path):
        self.cnngeo.load_weights(ckpt_path)
