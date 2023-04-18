import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk, ImageOps
from numpy import asarray
import numpy as np
import tensorflow as tf
import time

import clasify
import ploter

class Application:
    def __init__(self):
        try:
            self.modelClasifyer = tf.keras.models.load_model('testmodel')
        except:
            self.modelClasifyer = clasify.Model(10)

        self.plotter = ploter.Ploter()
        self.labels =['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.status = 'Model Loaded'
        self.my_w = tk.Tk()
        self.my_w.geometry("410x300")  # Size of the window 
        self.my_w.title('GUI')
        self.my_font1=('times', 18, 'bold')
        self.lable_1 = tk.Label(self.my_w,text='APPAREL IDENTIFIER',width=30,font=self.my_font1)  
        self.lable_1.grid(row=1,column=1,columnspan=4)
        self.lable_2 = tk.Label(self.my_w,text='Enter number of epochs:',width=50)  
        self.lable_2.grid(row=2,column=1,columnspan=5)
        
        e1=tk.IntVar()
        self.entry= tk.Entry(self.my_w,textvariable=e1, width= 8)
        self.entry.grid(row=2,column=3,columnspan=4)

        self.b1 = tk.Button(self.my_w, text='Train Model',
            width=20,command = lambda:self.train_model(e1.get()))
        self.b1.grid(row=3,column=1,columnspan=4)

        self.lable_3 = tk.Label(self.my_w,text=(self.status),width=50)
        self.lable_3.grid(row=4,column=1,columnspan=4)

        self.b2 = tk.Button(self.my_w, text='Predict Apparel',
            width=20,command = lambda:self.upload_file())
        self.b2.grid(row=5,column=1,columnspan=4)
        self.my_w.mainloop()  # Keep the window open

    def train_model(self,epochs):
        if epochs<= 0 or epochs>100:
            epochs = 10
        self.modelClasifyer = clasify.Model(epochs)
        self.modelClasifyer.model.save("testmodel")
        self.status = 'Completed'
        self.lable_3.config(text = self.status)

    def upload_file(self):
        f_types = [('All Files', '*.*'),
                    ('Jpg Files', '*.jpg'),
                    ('Jpeg Files', '*.JPEG'),
                    ('PNG Files','*.png')]
        # type of files to select
        filename = filedialog.askopenfilename(multiple=True,filetypes=f_types)
        self.clasification = ''
        col=2
        # start from column 1
        row=6
        # start from row 6
        for file in filename:
            img=Image.open(file) # read the image file
            img=img.resize((100,100)) # new width & height
            img=ImageTk.PhotoImage(img)
            lable_4 =tk.Label(self.my_w)
            lable_4.grid(row=row,column=col)
            lable_4.image = img # keep a reference! by attaching it to a widget attribute
            lable_4['image']=img # Show Image
            if(col==3): # start new line after third column
                row=row+1# start wtih next row
                col=1    # start with first column
            else:       # within the same row
                col=col+1 # increase to next column

            self.clasification += self.clasifyImage(file) + "   "

        lable_5 = tk.Label(self.my_w,text=(self.clasification),width=50)
        lable_5.grid(row=7,column=1,columnspan=4)

    def clasifyImage(self,filepath):
        print(filepath)
        with Image.open(filepath) as img:
            img.load()

        img = img.resize((28,28))
        img = ImageOps.grayscale(img)
        img = ImageOps.invert(img)

        test_image = np.array(img).astype('float32')/255
        test_image = np.expand_dims(test_image, axis=0)
        predictions = self.modelClasifyer.predict(test_image)

        print(predictions.shape)
        print(predictions[0])
        print(np.argmax(predictions[0]))
        print(self.labels[np.argmax(predictions[0])])
        self.plotter.plot_value_array(predictions[0])
        return self.labels[np.argmax(predictions[0])]

app = Application()
