import json
from data_loader import data_load
from matplotlib import pyplot as plt

with open("configs/mini_imagenet_fsl.json", "r") as file:
    data = json.load(file)

ds = data_load(['train'], data)
support, query = ds['train'].get_next_episode()
plt.imshow(support[2][0])
plt.show()
