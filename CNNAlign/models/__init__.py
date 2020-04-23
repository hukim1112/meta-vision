from .cnn_geo import CNN_geo
def load_model(model_name, backbone):
    if model_name == 'CNNgeo':
    	return CNN_geo()
