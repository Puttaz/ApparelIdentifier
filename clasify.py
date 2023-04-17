import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import dataset
import ploter
import math

BATCH_SIZE = 32

class Model:
    def __init__(self):

        self.data = dataset.Data('fashion_mnist',BATCH_SIZE)
        self.ploter = ploter.Ploter()

        # for image, label in data.test_dataset.take(1):
        #     break
        # image = image.numpy().reshape((28,28))

        # self.ploter.plotImage(image,plt.cm.binary)
        # self.ploter.plotImageSet(data.train_dataset,20,plt.cm.binary,data.class_names)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

        self.model.fit(self.data.train_dataset, epochs=20, steps_per_epoch=math.ceil(self.data.num_train_examples/BATCH_SIZE))

        test_loss, test_accuracy = self.model.evaluate(self.data.test_dataset, steps=math.ceil(self.data.num_test_examples/BATCH_SIZE))
        print('Accuracy on test dataset:', test_accuracy)
        print('Loss on test dataset:', test_loss)

        # for test_images, test_labels in self.data.test_dataset.take(1):
        #     test_images = test_images.numpy()
        #     test_labels = test_labels.numpy()
        #     predictions = self.model.predict(test_images)

        #     print(predictions.shape)
        #     print(predictions[0])
        #     print(np.argmax(predictions[0]))
        #     print(test_labels[0])

    def predict(self,image):
        return self.model.predict(image)


#__main__ function
if __name__ == "__main__":
    modelClasifyer = Model()
    modelClasifyer.model.save("testmodel")