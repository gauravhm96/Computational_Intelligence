# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:24:46 2022

@author: kirankumar athirala
"""

import sys
from tkinter import Tk, Button, Frame, Text, Label
from tkinter.scrolledtext import ScrolledText
import subprocess
from tkinter import filedialog
class PrintLogger(object):  # create file like object
    def __init__(self, textbox):  # pass reference to text widget
        self.textbox = textbox  # keep ref
    def write(self, text):
        self.textbox.configure(state="normal")  # make field editable
        self.textbox.insert("end", text)  # write text to textbox
        self.textbox.see("end")  # scroll to end
        self.textbox.configure(state="disabled")  # make field readonly
    def flush(self):  # needed for file like object
        pass
class MainGUI(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.root = Frame(self)
        self.root.pack()
        
        self.Label1 = Label(self.root, text="CI Model for Peek Detection and Peek Duration")
        self.Label1.pack()

        self.redirect_button = Button(self.root, text="Redirect console to widget", command=self.redirect_logging)
        self.redirect_button.pack()
        self.l1 = Label(self.root,  text='Input a value between 1 to 100 to choose one signal amongst signals:')
        self.l1.pack()
        self.Textbox1=Text(self.root, height=2, width=10)
        self.Textbox1.pack()
        self.train = Button(self.root, text="CLick_to_Train", command=self.train_peek_detection_model)
        self.train.pack()
        self.predict = Button(self.root, text="CLick_to_Predict", command=self.predict_peek_detection_model)
        self.predict.pack()
        self.log_widget = ScrolledText(self.root, height=20, width=120, font=("consolas", "8", "normal"))
        self.log_widget.pack()

    def redirect_logging(self):
        logger = PrintLogger(self.log_widget)
        sys.stdout = logger
        sys.stderr = logger


        
    def train_peek_detection_model(self):
        output = subprocess.run(['python', 'train.py'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = str(output.stdout, encoding='utf-8')
        self.log_widget.insert(1.0, output)
     
    def predict_peek_detection_model(self):
     
        process_data_file = filedialog.askopenfilename(initialdir = "/", 
                                   filetypes = (("csv files",
                                                 "*.csv*"),
                                                ("all files",
                                                 "*.*")),
                                   title = "Choose Training CSV File.")
        row_num = self.Textbox1.get("1.0")
        output = subprocess.run(['python', 'predict.py',"--input_file",process_data_file,"--row_num",row_num], stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = str(output.stdout, encoding='utf-8')
        self.log_widget.insert(1.0, output)

if __name__ == "__main__":
    app = MainGUI()
    app.mainloop()