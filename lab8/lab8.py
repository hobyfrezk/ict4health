# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:33:15 2017

@author: hoby
"""

low_risk = 0
medium_risk = 1
melanoma = 2

import cv2
import glob
import numpy as np

cv_img = {}

# loading all images and save them into a python list
for index, classification in enumerate(["low", "medium", "melanoma"]):
    cv_img[index] = []
    for img in glob.glob("images/"+classification+"*.jpg"):
        n= cv2.imread(img)
        [N1, N2, N3] = n.shape
        n= np.double(np.reshape(n,(N1*N2,N3)))
        cv_img[index].append(n)

N1 = N2 = N3 = classification = img = index = n = None
    
