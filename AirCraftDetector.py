# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:04:28 2021

@author: marco
"""

import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt

cap = cv.VideoCapture(r"C:\Users\marco\Desktop\Profissional\USP\Mestrado\ProcessamentoImagem\FinalProject\AirCraft-Detection\Dataset\EVision_VideoDataset\slow_traffic_small.mp4")
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
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

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
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]     
        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        
        
        img = cv.add(frame,mask)
 
        
        cv.imshow('Frame', img)
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        
        #Use 'q' in Keyboard to stop the video.
        if cv.waitKey(50) & 0xFF == ord('q'):
            break

        
cv.waitKey(0)
cv.destroyAllWindows()
