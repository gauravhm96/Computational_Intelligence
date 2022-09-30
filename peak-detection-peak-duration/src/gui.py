# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:00:37 2022

@author: kirankumar athirala
"""

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import sys
import os
import tkinter.scrolledtext as tkst
import subprocess
import re

root = Tk()
root.title("Peekdetection and peek duration")
root.geometry("875x450")
#root.configure(bg='#0c0c0c')
root.resizable(0,0)



def train_peek_detection_model():
     output = subprocess.run(['python', 'train.py'], stdout=subprocess.PIPE)
     output = str(output.stdout, encoding='utf-8')
     Textbox.insert(1.0, output)
     
def predict_peek_detection_model():
     
     process_data_file = filedialog.askopenfilename(initialdir = "/", 
                                   filetypes = (("csv files",
                                                 "*.csv*"),
                                                ("all files",
                                                 "*.*")),
                                   title = "Choose Training CSV File.")
     row_num = Textbox1.get("1.0",END)
     output = subprocess.run(['python', 'predict.py',"--input_file",process_data_file,"--row_num",row_num], stdout=subprocess.PIPE)
     output = str(output.stdout, encoding='utf-8')
     Textbox.insert(1.0, output)
     


Label1 = Label(root, text="CI Model for Peek Detection and Peek Duration")
l1 = Label(root,  text='Input a value between 1 to 100 to choose one signal amongst signals:')  # added one Label 

Button2 = Button(root, text="CLick_to_Train", command=train_peek_detection_model)  #)
Button3 = Button(root, text="CLick_to_Predict", command=predict_peek_detection_model)  #)
Textbox1=Text(root, height=2, width=10)
Textbox = tkst.ScrolledText(root, width=50, height=10)

Label1.grid(row=5, column=5, pady=15)
l1.grid(row=11,column=4) 
Textbox1.grid(row=11, column=5)
Button2.grid(row=12, column=5, pady=15)
Button3.grid(row=13, column=5, pady=15)
Textbox.grid(row=15, column=5)

root.mainloop()