from tkinter import *
import tkinter.filedialog as fdg
import tkinter.messagebox as msg
import os
import pandas as pd
from NBClassifier import NaiveBayes


# directory browse handler
def browseclick():
    dir_ent.delete(0, END)
    dir_ent.insert(0, fdg.askdirectory(mustexist=True))


# function for checking if all files exist in the given directory
def checkfiles():
    global path
    if (os.path.isfile(path + "\\structure.txt") and os.path.isfile(path + "\\test.csv") and os.path.isfile(
            path + "\\train.csv")):
        return TRUE
    return FALSE


def validate():
    global title, path
    if (not os.path.isdir(dir_ent.get())):
        msg.showerror(title, message="Please select a valid folder")
        return FALSE
    path = dir_ent.get()
    if (not checkfiles()):
        msg.showerror(title, message="Not all required files are present in the folder.")
        return FALSE
    if (not bin_ent.get().isdigit()):
        msg.showerror(title, message="Please submit a valid bins number")
        return FALSE
    return TRUE


# saving the results to output file
def saveResults(res):
    global path, title
    try:
        with open(path + "\\output.txt", "w+") as file:
            for k, v in res.items():
                file.write(str(k) + " " + str(v) + "\n")
    except Exception as e:
        msg.showerror(title, e.message)


# classification
def Classify():
    global cls, path, title
    # if model not yet trained...
    if not isinstance(cls, NaiveBayes) or path == "":
        msg.showerror(title, "Model not built yet")
        return
    try:
        df = pd.read_csv(path + "\\test.csv")  # read the test set
        res = cls.classify(df)  # classify
        saveResults(res)  # save results
        msg.showinfo(title, "Classification process finished. results saved at: " + path + "/output.txt")
    except Exception as e:
        msg.show(title, "Test file is invalid")


# build the model
def build():
    global cls, path, title
    if (not validate()):
        return
    try:  # read train set
        path = dir_ent.get()
        df = pd.read_csv(path + "\\train.csv")
        if df.shape[0] == 0:  # if its empty
            msg.showerror(title, "Invalid train file")
            return
        binsNumber = int(bin_ent.get())
        dd = getAttDict()  # get attribute to type dictionary
        cls = NaiveBayes(df, binsNumber, dd)  # intiate the classifier instance
        cls.train()  # train the model based on the train set
        msg.showinfo(title, "Building classifier using train-set is done!")
    except Exception as e:
        msg.showerror(title, "Invalid train file")


# gets attribut_name->type dictionary
def getAttDict():
    global path, title
    attDict = dict()
    try:
        with open(path + "\\Structure.txt", 'r') as f:
            for line in f.read().splitlines():
                split = line.split(" ")
                if split[2] == "NUMERIC":
                    attDict[split[1]] = split[2]
                else:  # if its noy numeric, put the possible nominal values as type
                    attDict[split[1]] = list(split[2][1:-1].split(","))
    except:
        msg.showerror(title, "Structure file is not valid")
    return attDict


def calculateAccuracy():
    global path
    counter = 0
    testSet = pd.read_csv(path + "\\test.csv")
    total = testSet.shape[0]
    with open(path + "\\output.txt") as f:
        lines = f.read().splitlines()
        for i, row in testSet.iterrows():
            if lines[i].split(' ')[1].lower() == str(row["class"]).lower():
                counter += 1
    acc = (float(counter) / total) * 100
    msg.showinfo(title, "Test accuracy: " + str(acc) + "%")


root = Tk()
title = "Naive Bayes Classifier"
root.wm_title(title)
root.geometry("280x130")
cls = None
dir_lbl = Label(root, text="Directory Path:")
bin_lbl = Label(root, text="Discretization Bins:")
dir_ent = Entry(root)
bin_ent = Entry(root)
browse_btn = Button(text="Browse", command=browseclick)
build_btn = Button(text="Build", command=build)
classify_btn = Button(text="Classify", command=Classify)
validate_btn = Button(text="Validate", command=calculateAccuracy)
validate_btn.grid(row=7, column=1)
dir_lbl.grid(row=3, sticky=E)
bin_lbl.grid(row=4, sticky=E)
dir_ent.grid(row=3, column=1)
bin_ent.grid(row=4, column=1)
browse_btn.grid(row=3, column=2)
build_btn.grid(row=5, column=1)
classify_btn.grid(row=6, column=1)
path = ""
root.mainloop()
