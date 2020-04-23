from matplotlib import pyplot as plt
def show_image(images):
    if isinstance(images, list):
        rows = len(images)
        shape = images[0].shape
        if len(shape) == 3:
            #list of single images
            cols = 1
            plot_images(images, (rows, cols))
        elif len(shape) == 4:
            #list of multiple images
            cols = shape[0]
            plot_images(images, (rows, cols))
        else:
            ValueError("The shape of images is wrong : {}".format(images[0].shape))
    else:
        TypeError("input must be list, but type is {}".format(type(images)))

def plot_images(images, plot_shape):
    rows, cols = plot_shape
    fig = plt.figure()
    for row in range(rows):
        for col in range(cols):
            fig.add_subplot(rows, cols, row*cols+col+1).imshow(images[row][col])
    plt.show()

