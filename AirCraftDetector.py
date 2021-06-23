# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:04:28 2021

@author: marco
"""

import cv2 as cv
import numpy as np
from msvcrt import getch
#import matplotlib.pyplot as plt

cap = cv.VideoCapture(r"C:\Users\marco\Desktop\Profissional\USP\Mestrado\ProcessamentoImagem\FinalProject\AirCraft-Detection\Dataset\EVision_VideoDataset\1FNB737.avi")
#Initialize ORB
orb = cv.ORB_create()


while(cap.isOpened()):
    #Read image
    ret, frame = cap.read()
    
    if ret == True:
        #Test ORB in this images.
        #Detect features with ORB
        kp = orb.detect(frame,None)
        kp, des = orb.compute(frame, kp)
        
        #Plot the Keypoints in the Original Image. Use Green color
        img_final = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
        
        reduced_frame = cv.resize(frame,(0,0), None, 0.5, 0.5)
        reduced_img_final = cv.resize(img_final,(0,0), None, 0.5, 0.5)
                
        img_stack = np.hstack((reduced_frame, reduced_img_final))
        
        cv.imshow('Stacked Image', img_stack)
        
        #Use 'q' in Keyboard to stop the video or Space (' ') to run step by step.
        Keyboard = cv.waitKey(1) & 0xFF
        if Keyboard == ord('q'):
            break
        elif Keyboard == ord(' '):
            cv.waitKey(0)
            continue
        
cv.destroyAllWindows()
