# CV-dogs
CV project in which I use yolov5+strongSORT+OSNet to detect my two dogs in the backyard 

![](https://github.com/Vol-v/CV-dogs/blob/main/dog1.gif)

![](https://github.com/Vol-v/CV-dogs/blob/main/dog2.gif)



# YOLOv5

For the model I used SOTA yolov5 which works great with this kind of task. In the fork of the yolo repo I uploaded an .mp4 example and also .ipynb in which I conducted training on my own custom dataset.  

# StrongSORT changes

The most noticeable changes I made to the algorithm are a all came from the defined task. I changed to total amount of detection per frame to 2 since I only have two dogs. I also made an additional check to yolo results after non max suppression: in case there were two detections of the same class on one frame, I changed a detection with lower probability to the oposite class. This way there were never two detections of the same dog on a frame.
