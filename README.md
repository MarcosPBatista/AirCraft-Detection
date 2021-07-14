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
it. However, since they are a company, it's not possible to know it in detail. The adjusted proposal is described as follow: 

![image](https://user-images.githubusercontent.com/85201876/125605439-8d80e70a-d973-4994-ad9d-fd66d8bfd0a3.png)

By step:

![image](https://user-images.githubusercontent.com/85201876/125605600-99868111-c5da-4133-bbbd-960fc03ed10e.png)

![image](https://user-images.githubusercontent.com/85201876/125605624-ea3e0abf-fd74-4c79-8be7-72a2f671d583.png)

![image](https://user-images.githubusercontent.com/85201876/125605652-9f715fc5-e4b0-4d50-bf6e-7ddfbb64327e.png)

![image](https://user-images.githubusercontent.com/85201876/125606900-f9c87c31-53b7-4fb5-9f57-3006487b9c51.png)


#4.1) Many applications use CNN in order to solve this problem. However, our approach here will be just to use Image Processing and Computer Vision Basic operations.

Final results:

- Airplane detected (With some background on the ground):

![image](https://user-images.githubusercontent.com/85201876/125605871-9a41fba6-7df3-43ff-8066-8b0bbbc66745.png)

![image](https://user-images.githubusercontent.com/85201876/125606005-3f81c339-74b3-4250-b8e1-461624cd367c.png)

![image](https://user-images.githubusercontent.com/85201876/125606073-5536d836-0b7b-47cc-96ec-7e336ee68c86.png)

![image](https://user-images.githubusercontent.com/85201876/125606147-316c9cd7-c51f-4b48-9641-21634af2b1be.png)

- With clouds, we have a lot of problems with background that could be isolated:

![image](https://user-images.githubusercontent.com/85201876/125606313-4eedb4d9-5ba3-496b-b0ed-0b60056e44cc.png)

![image](https://user-images.githubusercontent.com/85201876/125606387-4846a6a0-cd93-4052-afc0-4b92b47ff3cd.png)


Conclusions: 
- In all videos, it was possible to detect moving objects in the scene.
- It wasn’t able to filter background (clouds, etc).

Next steps:
- Use Optical Flow + IMU data to filter background;
- Use YOLO and Transfer Learning to detect Airplane.


