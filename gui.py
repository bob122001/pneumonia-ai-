from re import L
import tkinter as tk
from tkinter import filedialog
from unittest import result
import predict
from tkinter import *
from PIL import ImageTk, Image




window = tk.Tk()

window.geometry("500x500")

window.title("Pneumonia Diagnoses Tool")

l = Label(window, text = "", font=('Calibri 15 bold'))

l.pack()
   
def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    results = predict.predict_image(filename)
    print(filename)
    l["text"] = results
    print(results)
    


fileUpload = tk.Button(window, text="Upload File", command=UploadAction)
fileUpload.pack()




window.mainloop()






