# Object Detection with OpenCV

In this repo, I've worked on Object Detection with OpenCV, I've just aimed to get coordinates, width and height of object using traditional OpenCV algoritms, so this repo doesn't contain what that objects are. 
Firstly, I started with Edge Detection, Corner Detection and then Colorful Object Detection.

---

Here, I am going to explain some important topics:

1- [Object Detection with Color](https://github.com/ierolsen/Object-Detection-with-OpenCV/blob/main/4-object-detection-with-color.py)
2- [Watershed](https://github.com/ierolsen/Object-Detection-with-OpenCV/blob/main/7-watershed.ipynb)

You can find explanations below.

---

### Object Detection with Color

Using HSV color range which is determined as Lower and Upper, I detected colorful object. Here I prefered blue objects.

```python
# blue HSV
blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)
```
When I got the color range, I set capture size and then I read the capture.

First I apply Gaussian Blurring for decreasing the noises and details in capture. 
```python
#blur
blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0)
```
After Gaussian Blurring, I convert that into HSV color format.
```python
# HSV
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
```
To detect Blue Object, I define a mask.
```python
# mask for blue
mask = cv2.inRange(hsv, blueLower, blueUpper)
```
After mask, I have to clean around of masked object. Therefor I apply first Erosion and then Dilation
```python
# deleting noises which are in area of mask
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
```
After removing noises, the Contours have to be found
```python
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
center = None
```
If the Contours have been found, I'll get the biggest contour due to be well.
```python
# get max contour
c = max(contours, key=cv2.contourArea)
```
The Contours which are found have to be turned into rectangle deu to put rectangle their around. This cv2.minAreaRect() function returns a rectangle which is smallest to cover the area of object.
```python
rect = cv2.minAreaRect(c)
```
In the screen, I want to print the information of rectangle, therefor I need to reach its inform.
```python
((x,y), (width, height), rotation) = rect
s = f"x {np.round(x)}, y: {np.round(y)}, width: {np.round(width)}, height: {np.round(height)}, rotation: {np.round(rotation)}"
```
Using this rectangle I found, I want to get a Box. In the next, I will use this Box for drawing Rectangle.
```python
# box
box = cv2.boxPoints(rect)
box = np.int64(box)
```
Image Moment is a certain particular weighted average (moment) of the image pixels' intensities.
To find Momentum, I use Max. Contour named as "c". After that, I find Center point.
```python
# moment
M = cv2.moments(c)
center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
```
Now, I will draw the center which is found.
```python
# point in center
cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)
```
After Center Point, I draw Contour
```python
# draw contour
cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)
```
I want to print coordinators etc. in the screen 
```python
# print inform
cv2.putText(imgOriginal, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
```

And Final:

![detected_img](https://user-images.githubusercontent.com/30235603/100146802-bca98680-2e9a-11eb-9406-5e8a41b175df.png)

![detected2_img](https://user-images.githubusercontent.com/30235603/100215792-c0c4bb00-2f11-11eb-9c50-ab206fafbb62.png)

---

# Introduction to Watershed Algorithm

Previously, I worked on Object Detection algorithm and I make detected coordinaters, rotations and dimension of objects. This project is about Watershed. Basically Watershed algoritm is used for seperating the other objects in image. After apply this algorithm, I need to apply some Low-Pass Filtering and Morphological Operations deu to determine area and edge of objects.

### In Watershed Algoritm, I will apply this techniques:
 1- **Median Blurring**
 2- **Gray Scale**
 3- **Binary Threshold**
 4- **Opening**
 5- **Distance Transform**
 6- **Threshold for foreground**
 7- **Dilation for enlarging image to use for background**
 8- **Connected Components**
 9- **Watershed**
 10- **Find and Draw Contours around of objects**
 
I'll explain these subjects again when I explain code cells.

---

First of all I must import my libraries I'll use:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
After that, I need to read my image I want to apply Watershed:

```python
coin = cv2.imread("data/coins.jpg")

plt.figure(), plt.title("Original"), plt.imshow(coin), plt.axis("off");
```
![original](https://user-images.githubusercontent.com/30235603/100279106-38bcd080-2f66-11eb-8329-08939450da8f.png)


By the way, the semicolon (;) which is in end of **plt.axis("off")** is a trick for ingoring some inputs of matplotlib.


To remove the noises which are on image I will apply one of Low-Pass Filtering methods called **MedianBlurring**

```python
coin_blur = cv2.medianBlur(src=coin, ksize=13)

plt.figure(), plt.title("Low Pass Filtering (Blurring)"), plt.imshow(coin_blur), plt.axis("off");
```
![Low Pass Filtering (Blurring)](https://user-images.githubusercontent.com/30235603/100279112-39edfd80-2f66-11eb-8d3d-369f2a358e99.png)

After that, I need to **convert** color of image to **Gray**
```python
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)

plt.figure(), plt.title("Gray Scale"), plt.imshow(coin_gray, cmap="gray"), plt.axis("off");
```
![Gray Scale](https://user-images.githubusercontent.com/30235603/100279114-39edfd80-2f66-11eb-85e7-9d72fedf6360.png)


Using **Binary Threshold** I will make image specific between coins and background
```python
ret, coin_thres = cv2.threshold(src=coin_gray, thresh=75, maxval=255, type=cv2.THRESH_BINARY)

plt.figure(), plt.title("Binary Threshold"), plt.imshow(coin_thres, cmap="gray"), plt.axis("off");
```
![Binary Threshold](https://user-images.githubusercontent.com/30235603/100279115-3a869400-2f66-11eb-9132-2d990dbc30be.png)

As you can see after **Thresholding** it is almostly clear between coins and background.

Now, I will try to draw these **Contours** 
```python
contour, hierarchy = cv2.findContours(image=coin_thres.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contour)):
    
    if hierarchy[0][i][3] == -1: # external contour
        cv2.drawContours(image=coin,contours=contour,contourIdx=i, color=(0,255,0), thickness=10)
        
plt.figure(figsize=(7,7)), plt.title("After Contour"), plt.imshow(coin, cmap="gray"), 
plt.axis("off");
```
![After Contour1](https://user-images.githubusercontent.com/30235603/100279116-3a869400-2f66-11eb-962d-f0632c81d255.png)
###### But as you can see it doesn't work, I couldn't seperate their edge. 
I will use other method called **Watershed** for seperating and draw their edge. First method is not **Watershed**

---

Before Watershed, I need to apply all of this techniques. For that:
```python
# read data
coin = cv2.imread("data/coins.jpg")
plt.figure(), plt.title("Original"), plt.imshow(coin), plt.axis("off");

# Blurring
coin_blur = cv2.medianBlur(src=coin, ksize=15)
plt.figure(), plt.title("Low Pass Filtering (Blurring)"), plt.imshow(coin_blur), plt.axis("off");

# Gray Scale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.title("Gray Scale"), plt.imshow(coin_gray, cmap="gray"), plt.axis("off");

# Binary Threshold
ret, coin_thres = cv2.threshold(src=coin_gray, thresh=65, maxval=255, type=cv2.THRESH_BINARY)
plt.figure(), plt.title("Binary Threshold"), plt.imshow(coin_thres, cmap="gray"), 
plt.axis("off");
```
![original](https://user-images.githubusercontent.com/30235603/100279106-38bcd080-2f66-11eb-8329-08939450da8f.png)
![Low Pass Filtering (Blurring)](https://user-images.githubusercontent.com/30235603/100279112-39edfd80-2f66-11eb-8d3d-369f2a358e99.png)
![Gray Scale](https://user-images.githubusercontent.com/30235603/100279114-39edfd80-2f66-11eb-85e7-9d72fedf6360.png)
![Binary Threshold](https://user-images.githubusercontent.com/30235603/100279115-3a869400-2f66-11eb-9132-2d990dbc30be.png)

First I need to remove connetion between coins. Almostly every coins connetc other self, therefor using Opening from **Morphological Operations**, I will **Open** them

```python
kernel = np.ones((3,3), np.uint8)

opening = cv2.morphologyEx(coin_thres, cv2.MORPH_OPEN, kernel=kernel, iterations=2)

plt.figure(), plt.title("Opening"), plt.imshow(opening, cmap="gray"), plt.axis("off");
```
![Opening](https://user-images.githubusercontent.com/30235603/100279119-3b1f2a80-2f66-11eb-8e78-8b18708fa7c6.png)


To romevo the connection of between coins I will use **Distance Transform**.
After that I can see **Distance** between objects (coins)
```python
dist_transform = cv2.distanceTransform(src=opening, distanceType=cv2.DIST_L2, maskSize=5)

plt.figure(), plt.title("Distance Transform"), plt.imshow(dist_transform, cmap="gray"), plt.axis("off");
```
![Distance Transform](https://user-images.githubusercontent.com/30235603/100279121-3b1f2a80-2f66-11eb-9b09-7e5ee542610e.png)


After finding distance, to find image which is in foreground what it is, I'll minimizing that using **Threshold**.
```python
ret, sure_foreground = cv2.threshold(src=dist_transform, thresh=0.4*np.max(dist_transform), maxval=255, type=0)

plt.figure(), plt.title("Fore Ground"), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off");
```
![Fore Ground](https://user-images.githubusercontent.com/30235603/100279124-3bb7c100-2f66-11eb-9c55-b06f5d198d84.png)

To find background what it is, I will enlarging image using **Dilate**.
```python
sure_background = cv2.dilate(src=opening, kernel=kernel, iterations=1) #int

sure_foreground = np.uint8(sure_foreground) # change its format to int
```
And now, I can **Subtrack** them eachothers **(BackGround - ForeGround)** so that, the image can be more understandable. Here is the result of **Opened** and **Dilated** image
```python
unknown = cv2.subtract(sure_background, sure_foreground)

plt.figure(), plt.title("BackGround - ForeGround = "), plt.imshow(unknown, cmap="gray"), plt.axis("off");
```
![BackGround - ForeGround =](https://user-images.githubusercontent.com/30235603/100279126-3bb7c100-2f66-11eb-9336-a784a25cd5d5.png)

After these steps, I need to find **Markers** for giving inputs for Watershed algorithm. And now I'll provide **Connection** between **Components**.

```python
ret, marker = cv2.connectedComponents(sure_foreground)

marker = marker + 1

marker[unknown == 255] = 0 # White area is turned into Black to find island for watershed

plt.figure(), plt.title("Connection"), plt.imshow(marker, cmap="gray"), plt.axis("off");
```
![Connection](https://user-images.githubusercontent.com/30235603/100279128-3bb7c100-2f66-11eb-9aa2-fb60e2bc7d25.png)


After that, now I can apply **Watershed Algorithm** and I can make **segmentation**
```python
marker = cv2.watershed(image=coin, markers=marker)

plt.figure(), plt.title("Watershed"), plt.imshow(marker, cmap="gray"), plt.axis("off");
```
![Watershed](https://user-images.githubusercontent.com/30235603/100279129-3c505780-2f66-11eb-8160-c8a3ef11c402.png)


As a last step, I'll find and **Draw** **Contours** around of Coins.
```python
contour, hierarchy = cv2.findContours(image=marker.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contour)):
    
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image=coin,contours=contour,contourIdx=i, color=(255,0,0), thickness=3)
        
plt.figure(figsize=(7,7)), plt.title("After Contour"), plt.imshow(coin, cmap="gray"), plt.axis("off");
```
![After Contour2](https://user-images.githubusercontent.com/30235603/100279130-3c505780-2f66-11eb-813f-e14adc119f3a.png)

---

