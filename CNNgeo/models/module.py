import numpy as np
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import tensorflow as tf
import tensorflow_addons as tfa

def feature_l2_normalization(feature_map):
    #input(feature map) has tensor-shape (N, H, W, D)
    l2_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(feature_map, 2), axis=-1))  # (N, H, W)
    l2_norm = tf.expand_dims(l2_norm, axis=-1)  # (N, H, W, 1)
    # (N, H, W, D) tensor divided by (N, H, W, 1) tensor
    return feature_map / (l2_norm+1E-5)

# repeat penalty
def normalize_correlation(corr_score):
    ambiguous_match_penalty = tf.math.sqrt(tf.reduce_sum(tf.pow(corr_score, 2), axis=[3, 4], keepdims=True))
    corr_score = tf.math.divide(corr_score, ambiguous_match_penalty+0.00001)
    # see eq (2) in "End-to-end weakly-supervised semantic alignment"
    return corr_score

def generate_inlier_mask(geo_parameters, geo_transform, map_size):
    batch_size=len(geo_parameters)
    height, width = map_size
    identity_mask = np.zeros((height*width*height*width))
    idx_list = list(range(0, height*width*height*width, height*width+1))
    identity_mask[idx_list] = 1
    identity_mask = np.reshape(identity_mask, (height,width,height,width))
    dilation_filter = generate_binary_structure(2, 2)
    for h in range(height):
        for w in range(width):
            identity_mask[h,w] = binary_dilation(identity_mask[h,w], structure=dilation_filter).astype(identity_mask.dtype)
    identity_mask = identity_mask[tf.newaxis,::]
    identity_mask = np.reshape(identity_mask, (1, height,width,height,width))

    identity_mask = tf.constant(identity_mask, tf.float32)
    identity_mask = tf.tile(identity_mask, [batch_size,1,1,1,1]) #[BN,H,W,H,W] of identity matrix
    '''
        We reshape it for convenient parallel processing of remapping.
        Each H*W inlier masks are remaped by the same geometric transformation.
        Therefore, We calculate BN remaps and repeat H*W of it.
    '''
    identity_mask = tf.reshape(identity_mask, [batch_size*height*width, height, width, 1]) #[BN*H*W,H,W,1].

    #calculate estimated coordinates of source grids on target feature grids
    control_points = tf.constant([[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
                                       [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                                       [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32)  
    control_points = tf.tile(control_points[tf.newaxis,::], [batch_size, 1, 1]) # [BN, 9, 2]
    x_s, y_s = geo_transform(control_points, -geo_parameters, (height,width))
    #calculate BN remaps
    remaps = tf.stack([x_s, y_s], axis=-1) #[BN,H,W,2]
    #repeat each remap H*W times. 
    remaps = tf.tile(remaps[:,tf.newaxis,::], [1,height*width, 1, 1, 1]) #[BN,H*W,H,W,2]
    print(remaps.shape)
    remaps = tf.reshape(remaps, [batch_size*height*width,height,width,2]) #[BN*H*W,H,W,2]

    inlier_masks = tfa.image.resampler(identity_mask, remaps) #inputs <= identity_mask([BN*H*W,H,W,1]) remaps([BN*H*W,H,W,2])
    inlier_masks = tf.reshape(inlier_masks, [batch_size, height, width, height, width]) #reshape again. 
    return inlier_masks #[BN,H,W,H,W]
