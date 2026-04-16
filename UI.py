from tkinter import *
from tkinterdnd2 import DND_FILES, TkinterDnD
    
def drop(event):
    file_path = event.data
    print("Dropped file:", file_path)
    startbutton = Button(root, text="Start", width=25, command=start)
    startbutton.pack()

def start(data):
    #turn img to data with hog (function)
    #use that data to run model (funciton)
    #output from model diterimens if the window says cat or dog (gui function)
    #if confidecenc is or close to boarder pop up a window that says "was i right" and flag the anweser and result in csv (gui funciton)
    pass
root = TkinterDnD.Tk()
label = Label(root, text="Drag file here", bg="lightgray")
label.pack(expand=True, fill="both", padx=10, pady=10)

label.drop_target_register(DND_FILES)
label.dnd_bind("<<Drop>>", drop)

quitbutton = Button(root, text="Quit", width=25, command=root.destroy)


label.pack()

quitbutton.pack()

root.geometry("750x750")
root.mainloop()
