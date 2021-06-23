# AirCraft-Detection
This project is related to a course taught by Moacir regarding Process Image in ICMC-USP.
Student Name: Marcos Paulo Batista (8549921 - marcos.batista@usp.br)

#1) Motivation of this Project
UAVs (Unmanned Aircraft Vehicle) are used in many applications, such as Agriculture (Identify weak plants and problems),
Marketing/Multimedia (recording video), Surveillance (in order to identify frontiers) and so on. In almost all of those aplications,
it is used the VLOS (View Line of Sight). However there are applications that require BVLOS (Beyond Line Of Sight) Flights,
such as Logistic, Goods Transportation and Taxing. In this scenario, there will be a lot of UAVs and Manned Aircraft in the skies.
USA and Euroupe are already discussing about regulations on it, since a UAV can crash with a manned aircraft and someone can be damaged.
One specification of it is to identify a manned aircraft in a radius of 1.2 km which is quite far and challenging. One way to do this is
to use Camera (Iris Automation use it). In this way this project is focused in detecting manned aircraft in the dataset described bellow.

#2)The dataset used is https://www.kaggle.com/adriancarrio/aircraft-detection-and-sky-segmentation-dataset. The paper is:
"Carrio, A., Fu, C., Collumeau, JF. et al. SIGS: Synthetic Imagery Generating Software for the Development and Evaluation 
of Vision-based Sense-And-Avoid Systems. J Intell Robot Syst 84, 559–574 (2016). https://doi.org/10.1007/s10846-015-0286-z".
It uses synthetic images in order to create dataset since this kind of dataset is quite expensive. There are videos with 
synthetic airplane in it.

#3) There are many possible ways to do that. Iris Automation (https://www.irisonboard.com/casia/) uses machine learning to detect and classify
it. However, since they are a company, it's not possible to know it in detail. One approach which is suitable for this scope/time is to use
motion detection in order to identify possible objects. One technique is Optical Flow. We can detect this points, cluster them, maybe apply some 
morphological in those cluster, filter and track them. In order to track, a feature detector/descritor could be ORB which is fast than SURF/SIFT
and is invariant to rotation and scale.

Consideration: Since Optical Flow and ORB aren't that easy to implement by hand. The focus here will be to use OpenCV and try to adjust the implementation
to achieve good results.

#4) Up to now it run ORB to detect features as shown bellow.
![image](https://user-images.githubusercontent.com/85201876/123179368-a4650e80-d45f-11eb-94d6-ec2a9314a543.png)

