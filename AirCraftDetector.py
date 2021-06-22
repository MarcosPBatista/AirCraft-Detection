# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:04:28 2021

@author: marco
"""

import cv2 as cv
#import numpy as np
#import matplotlib.pyplot as plt

cap = cv.VideoCapture(r"C:\Users\marco\Desktop\Profissional\USP\Mestrado\ProcessamentoImagem\FinalProject\AirCraft-Detection\Dataset\EVision_VideoDataset\1FNB737.avi")
#Initialize ORB
orb = cv.ORB_create()

#Initialize Optical Flow Parameters
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


while(cap.isOpened()):
    #Read image
    ret, frame = cap.read()
    
    if ret == True:
        #Test ORB in this images.
# =============================================================================
#         #Detect features with ORB
#         kp = orb.detect(frame,None)
#         kp, des = orb.compute(frame, kp)
#         
#         #Plot the Keypoints in the Original Image. Use Green color
#         img_final = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
# =============================================================================
        
        
        cv.imshow('Frame',img_final)
        
        #Use 'q' in Keyboard to stop the video.
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
        
cv.waitKey(0)
cv.destroyAllWindows()
