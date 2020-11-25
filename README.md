# Object Detection with OpenCV

In this repo, I've worked on Object Detection with OpenCV, I've just aimed to get coordinators, width and height of object using traditional OpenCV algoritms, so this repo doesn't contain what that objects are. 
Firstly, I started with Edge Detection, Corner Detection and then Colorful Object Detection.

---

### Notebook 4 (Object Detection with Color)

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

![detected2_img](https://user-images.githubusercontent.com/30235603/100214408-257f1600-2f10-11eb-871e-743ead2e50b0.png)

___
