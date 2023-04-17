import numpy as np
import matplotlib.pyplot as plt
import math

class Ploter:
    def __init__(self):
        pass

    def plotImage(self,image,colormap):
        plt.figure()
        plt.imshow(image, cmap=colormap)
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def plotImageSet(self,dataset,size,colormap,classnames):
        plt.figure(figsize=(10,10))
        for i, (image, label) in enumerate(dataset.take(size)):
            image = image.numpy().reshape((28,28))
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap=colormap)
            plt.xlabel(classnames[label])
        plt.show()
