# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:04:28 2021

@author: marco
"""

import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt

cap = cv.VideoCapture(r"C:\Users\marco\Desktop\Profissional\USP\Mestrado\ProcessamentoImagem\FinalProject\AirCraft-Detection\Dataset\EVision_VideoDataset\1FNC172.aviq")
#Initialize ORB
orb = cv.ORB_create()

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(cap.isOpened()):
    #Read image
    ret, frame2 = cap.read()
    
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
        
        
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        prvs = next
 
        
        cv.imshow('Frame', bgr)
        
        # Now update the previous frame and previous points

        
        #Use 'q' in Keyboard to stop the video.
        if cv.waitKey(50) & 0xFF == ord('q'):
            break

        
cv.waitKey(0)
cv.destroyAllWindows()
