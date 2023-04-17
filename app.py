import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk, ImageOps
from numpy import asarray
import numpy as np
import tensorflow as tf

import clasify

class Application:
    def __init__(self):
        self.modelClasifyer = tf.keras.models.load_model('testmodel')
        #modelClasifyer = clasify.Model()
        #modelClasifyer.model.save("testmodel")
        self.labels =['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.my_w = tk.Tk()
        self.my_w.geometry("410x300")  # Size of the window 
        self.my_w.title('GUI')
        self.my_font1=('times', 18, 'bold')
        self.l1 = tk.Label(self.my_w,text='Upload Files & display',width=30,font=self.my_font1)  
        self.l1.grid(row=1,column=1,columnspan=4)
        self.b1 = tk.Button(self.my_w, text='Upload Files',
            width=20,command = lambda:self.upload_file())
        self.b1.grid(row=2,column=1,columnspan=4)
        self.my_w.mainloop()  # Keep the window open

    def upload_file(self):
        f_types = [('All Files', '*.*'),
                    ('Jpg Files', '*.jpg'),
                    ('Jpeg Files', '*.JPEG'),
                    ('PNG Files','*.png')]   # type of files to select 
        filename = filedialog.askopenfilename(multiple=True,filetypes=f_types)
        self.clasification = ''
        col=1 # start from column 1
        row=3 # start from row 3 
        for f in filename:
            img=Image.open(f) # read the image file
            img=img.resize((100,100)) # new width & height
            img=ImageTk.PhotoImage(img)
            e1 =tk.Label(self.my_w)
            e1.grid(row=row,column=col)
            e1.image = img # keep a reference! by attaching it to a widget attribute
            e1['image']=img # Show Image
            if(col==3): # start new line after third column
                row=row+1# start wtih next row
                col=1    # start with first column
            else:       # within the same row
                col=col+1 # increase to next column

            self.clasification += self.clasifyImage(f) + "   "

        l6 = tk.Label(self.my_w,text=(self.clasification),width=50)
        l6.grid(row=4,column=1,columnspan=4)
        
    def clasifyImage(self,filepath):
        #load the image file
        print(filepath)
        # filepath = "./images/testT.jpg"
        with Image.open(filepath) as img:
            img.load()

        img = img.resize((28,28))
        img = ImageOps.grayscale(img)
        img = ImageOps.invert(img)

        # img.show()

        #test_image = asarray(img)
        test_image = np.array(img).astype('float32')/255
        test_image = np.expand_dims(test_image, axis=0)
        predictions = self.modelClasifyer.predict(test_image)

        print(predictions.shape)
        print(predictions[0])
        print(np.argmax(predictions[0]))
        print(self.labels[np.argmax(predictions[0])])
        return self.labels[np.argmax(predictions[0])]

app = Application()
