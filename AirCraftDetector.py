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

#Initialize Matcher and Previous Image/Keypoint/Descriptors
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
ret,last_frame = cap.read()
kp = orb.detect(last_frame,None)
last_kp, last_des = orb.compute(last_frame, kp)


while(cap.isOpened()):
    #Read image
    ret, frame = cap.read()
    
    if ret == True:
        #Detect features with ORB
        kp = orb.detect(frame,None)
        kp, des = orb.compute(frame, kp)
        
        #Plot the Keypoints in the Original Image. Use Green color
        ORB_raw = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
        
# =============================================================================
#         ########################################################
#         #Resize and stack in order to get easier visualization
#         resize_frame = cv.resize(frame,(0,0), None, 0.5, 0.5)
#         resize_ORB_Raw = cv.resize(ORB_raw,(0,0), None, 0.5, 0.5)
#         img_stack = np.hstack((resize_frame, resize_ORB_Raw))
#         
#         cv.imshow('Stacked Image', img_stack)
#         ########################################################
# =============================================================================
        
        #Check Matching between last image and actual using ORB
        matches = matcher.match(last_des, des)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        match_img = cv.drawMatches(last_frame, last_kp,  
                                   frame,kp, matches[:30],None, flags = 2) 
        
        #Create binary image using ORB Keypoints
        ORB_bin = np.zeros(frame.shape, dtype = np.uint8)
        for i in range(len(kp)):
            ORB_bin[int(kp[i].pt[1]),int(kp[i].pt[0])] = 255
        
        #Resize and show images
        reduced_ORB_bin = cv.resize(ORB_bin,(0,0), None, 0.5, 0.5)
        reduced_match_img = cv.resize(match_img,(0,0), None, 0.5, 0.5)
        resize_ORB_Raw = cv.resize(ORB_raw,(0,0), None, 0.5, 0.5)
        cv.imshow('ORB_bin', np.hstack((resize_ORB_Raw, reduced_ORB_bin)))
        cv.imshow('Match between previous and actual image', reduced_match_img)
        
        #Update last detections and frame
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
