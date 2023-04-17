from PIL import Image
from PIL import ImageOps
from numpy import asarray
import numpy as np
import tensorflow as tf

import clasify

modelClasifyer = tf.keras.models.load_model('testmodel')
labels =['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#modelClasifyer = clasify.Model()
#modelClasifyer.model.save("testmodel.h5")

#load the image file
filepath = "./images/testT.jpg"
with Image.open(filepath) as img:
    img.load()

img = img.resize((28,28))
img = ImageOps.grayscale(img)
img = ImageOps.invert(img)

img.show()

#test_image = asarray(img)
test_image = np.array(img).astype('float32')/255
test_image = np.expand_dims(test_image, axis=0)
predictions = modelClasifyer.predict(test_image)

print(predictions.shape)
print(predictions[0])
print(np.argmax(predictions[0]))
print(labels[np.argmax(predictions[0])])