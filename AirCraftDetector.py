# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:04:28 2021

@author: marco
"""

import cv2 as cv
import numpy as np

erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'

def ORB_detect(Img, orb):
    
    #Detect features with ORB
    kp = orb.detect(Img,None)
    kp, des = orb.compute(Img, kp)
    
    #Plot the Keypoints in the Original Image. Use Green color
    ORB_img_raw = cv.drawKeypoints(Img, kp, None, color=(0,255,0), flags=0)
    
    return kp, des, ORB_img_raw

def Resize_Img2Plot(img1, img2, img3, factor = 0.5):
    
    resized_img1 = cv.resize(img1,(0,0), None, factor, factor)
    resized_img2 = cv.resize(img2,(0,0), None, factor, factor)
    resized_img3 = cv.resize(img3,(0,0), None, factor, factor)
    
    return resized_img1, resized_img2, resized_img3

def MatchImgs(frame, kp, des, last_frame, last_kp, last_des):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    #Check Matching between last image and actual using ORB
    matches = matcher.match(last_des, des)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    match_img = cv.drawMatches(last_frame, last_kp,  
                                   frame,kp, matches[:30],None, flags = 2) 
    
    return match_img
def main():
    cap = cv.VideoCapture(r"C:\Users\marco\Desktop\Profissional\USP\Mestrado\ProcessamentoImagem\FinalProject\AirCraft-Detection\Dataset\EVision_VideoDataset\2FNB737.avi")
    #Initialize ORB
    orb = cv.ORB_create()
    
    #Initialize Matcher and Previous Image/Keypoint/Descriptors
    ret,frame = cap.read()
    last_frame = frame #cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp = orb.detect(last_frame,None)
    last_kp, last_des = orb.compute(last_frame, kp)
    ORB_bin = np.zeros(last_frame.shape, dtype = np.uint8)
    last_ORB_bin = ORB_bin
    
    while(cap.isOpened()):
        #Read image
        ret, frame_raw = cap.read()
        
        if ret == True:
            
            
            frame = frame_raw#cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)
            
            #Detect keypoints using ORB
            kp, des, ORB_raw = ORB_detect(frame, orb)
            
            #Match frame and last_frame using ORB
            match_img = MatchImgs(frame, kp, des, last_frame, last_kp, last_des)
            ORB_bin = np.zeros(last_frame.shape, dtype = np.uint8)
            
            #Convert ORB to Binary Image
            for i in range(len(kp)):
                ORB_bin[int(kp[i].pt[1]),int(kp[i].pt[0])] = 255
            
            #Apply Gaussian Filter to eliminate punctual kp
            ORB_bin = cv.GaussianBlur(ORB_bin, (3,3), 1)
            
            #Apply Dilation to enhance
            kernel = np.ones((8,8))
            ORB_bin = cv.dilate(ORB_bin, kernel)
    
            #Calculate Movement
            ORB_bin_diff = cv.subtract(ORB_bin, last_ORB_bin)
            
            cv.imshow('ORB_bin_diff', cv.resize(ORB_bin_diff, (0,0), fx=0.5, fy=0.5))
            
            #Resize and show images
            reduced_ORB_bin, reduced_match_img, resize_ORB_Raw = Resize_Img2Plot(
                ORB_bin,match_img, ORB_raw)
            
            cv.imshow('ORB_bin', np.hstack((resize_ORB_Raw, reduced_ORB_bin)))
            cv.imshow('Match between previous and actual image', reduced_match_img)
            
            #Update last detections and frame
            last_frame, last_ORB_bin = frame, ORB_bin
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

if __name__ == "__main__":
    main()