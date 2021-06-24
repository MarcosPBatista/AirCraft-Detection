# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:04:28 2021

@author: marco
"""

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(r"C:\Users\marco\Desktop\Profissional\USP\Mestrado\ProcessamentoImagem\FinalProject\AirCraft-Detection\Dataset\EVision_VideoDataset\1FNB737.avi")
#Initialize ORB
orb = cv.ORB_create()
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

ret,last_frame = cap.read()
kp = orb.detect(last_frame,None)
last_kp, last_des = orb.compute(last_frame, kp)


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
        
        #Check Matching between last image and actual
        matches = matcher.match(last_des, des)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        match_img = cv.drawMatches(last_frame, last_kp,  
                                   frame,kp, matches[:30],None, flags = 2) 
        reduced_match_img = cv.resize(match_img,(0,0), None, 0.5, 0.5)
        bin_ORB = np.zeros(frame.shape, dtype = np.uint8)
        
        for i in range(len(kp)):
            bin_ORB[int(kp[i].pt[1]),int(kp[i].pt[0])] = 255
        reduced_bin_ORB = cv.resize(bin_ORB,(0,0), None, 0.5, 0.5)
            
        cv.imshow('Bin_ORB', np.hstack((reduced_img_final, reduced_bin_ORB)))
        
        cv.imshow('Match between previous and actual image', reduced_match_img)
        
        last_frame = frame
        last_kp, last_des = kp, des
        #Use 'q' in Keyboard to stop the video or Space (' ') to run step by step.
        Keyboard = cv.waitKey(1) & 0xFF
        if Keyboard == ord('q'):
            break
        elif Keyboard == ord(' '):
            cv.waitKey(0)
            continue
    else:
        break
    
        
cv.destroyAllWindows()
