# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:24:40 2021

@author: 1806638
"""



import glob
import os
import time 
import cv2
import json 




def crop_from_dic(dic):
    name = dic['name']
    bndbox =[ dic['bndbox']['xmin'] ,
             dic['bndbox']['ymin'] ,
             dic['bndbox']['xmax'],
             dic['bndbox']['ymax']
             ]
    return name, bndbox

files = glob.glob(r"train\*.json")

with open("train.csv",'a') as file:
    file.write("filename," + "width," + "height,"+"class,"+"xmin,"+"ymin,"+"xmax,"+"ymax\n")


for fCounter , f in enumerate(files):
    try:
        fname = f.split(".")[0]
        fname = fname.replace("\\","^").split("^")[-1]
        print(fname)
        print(f"{fCounter} done so far!")
        with open(f, 'r') as f:
            data = json.loads(f.read())
            
        cntr1 = 0
        cntr2 = 0
        for d in data['objects']:
            #print(d)
            tag , lst = crop_from_dic(d)
            #print(tag)
            im = cv2.imread(os.path.join("train\\" , fname )+ ".png")
            height, width, _ = im.shape
            h,w,_ = im.shape
            xmin, ymin , xmax , ymax = lst

            
            if "parcel_" in tag:
                with open("train.csv",'a') as file:
                    file.write(fname + ".png," + str(w) +"," + str(h) + "," + 
                               "parcel,"+ str(xmin) + ","+ str(ymin) + ","+
                                str(xmax) +","+ str(ymax) +"\n")
                    
            elif "reflection_" in tag or "nonParcel_" in tag:
                with open("train.csv",'a') as file:
                    file.write(fname + ".png," + str(w) +"," + str(h) + "," + 
                               "nonParcel,"+ str(xmin) + ","+ str(ymin) + ","+
                                str(xmax) +","+ str(ymax) +"\n")
            
            
                
                
        
        print("+++++++++++++++++++++++++++++++++++++++++=")
    except:
        print("Skipped !")





    
