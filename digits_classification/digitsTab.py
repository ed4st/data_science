from tkinter import *
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ipywidgets import Output
from IPython.display import display, clear_output
out = Output()


class Predict(Frame):
    """Handwritten digits prediction class"""
    def __init__(self, parent, fitted_models):
            self.multivariate_model = fitted_models[0]
            self.lda_model = fitted_models[1]
            self.qda_model = fitted_models[2]
            self.log_model = fitted_models[3]
            
            Frame.__init__(self, parent)
            self.parent = parent
            self.color = "black"
            self.brush_size = 20
            self.img_width = 28
            self.img_height = 28
            self.setUI()
            
            
    def set_color(self, new_color):
            """Additional brush color change"""
            self.color = new_color

    def draw(self, event):
            """Method to draw"""
            self.canv.create_oval(event.x - self.brush_size, event.y - self.brush_size,
                                  event.x + self.brush_size, event.y + self.brush_size,
                                  fill=self.color, outline=self.color)
    def predict(self):
            """
            Save the current canvas state 
            and predict the handwritten digit
            """
            self.canv.update()
            ps = self.canv.postscript(colormode='mono')
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            img.save('result.png')
            x = Predict.transform_image(self)
            
            #prediction with multivariate regression
            Y_hat_test = self.multivariate_model.predict([x])
            C_multivariate = map(np.argmax, Y_hat_test)  # classification vector
            C_multivariate = list(C_multivariate)
            multivariate_predict = C_multivariate[0]

            
            #prediction with Linear Discriminant Analysis (LDA)
            lda_predict = self.lda_model.predict([x])[0]
            qda_predict = self.qda_model.predict([x])[0]
            log_predict = self.log_model.predict([x])[0]
            
            baseline_label = Label(self, text='Baseline: ' + str(multivariate_predict) )
            baseline_label.grid(row=0, column=1, padx=5, pady=5)
            lda_label = Label(self, text=' LDA: '+ str(lda_predict))
            lda_label.grid(row=0, column=2, padx=5, pady=5)
            qda_label = Label(self, text='QDA: '+ str(qda_predict))
            qda_label.grid(row=1, column=1, padx=5, pady=5)
            log_label = Label(self, text=' Logistic: '+str(log_predict))
            log_label.grid(row=1, column=2, padx=5, pady=5)

    @staticmethod
    def transform_image(self):
        """
        Process the input digit image and returns a resized 28x28 image 
        """
        im = cv2.imread("result.png", 0)
        im2 = cv2.resize(im, (28, 28))
        im = im2.reshape(28, 28, -1)
        im = im.reshape(1, 1, 28, 28)
        im = cv2.bitwise_not(im)
        im = im.reshape(28,28)
        
        with out:
            clear_output()
        
        # resize
        img = np.array(im)
        img = img.reshape(28*28,)
        
        #img = img/255.0
        
        return img
        
        
    def setUI(self):
            """Setup for all UI elements"""
            self.parent.title("Handwritten digits classification")
            self.pack(fill=BOTH, expand=1)
            self.columnconfigure(6,weight=1)
            self.rowconfigure(2, weight=1)
            self.canv = Canvas(self, bg="white")
            self.canv.grid(row=2, column=0, columnspan=7,
                           padx=5, pady=5,
                           sticky=E + W + S + N)
            self.canv.bind("<B1-Motion>",
                           self.draw)
			
			
            #size_lab = Label(self, text="Classificator: ")
            #size_lab.grid(row=0, column=0, padx=5)
            predict_btn = Button(self, text="Predict", width=10, command=lambda: self.predict())
            predict_btn.grid(row=0, column=0)
            delete_btn = Button(self, text="Clear", width=10, command=lambda: self.canv.delete("all"))
            delete_btn.grid(row=1, column=0, sticky=W)