import cv2
import matplotlib.pyplot as plt
from tensorflow import *
import os
import sys

#os.chdir(os.path.dirname(sys.argv[0]))
cwd = os.getcwd()

config = cwd + r'\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model = cwd + r'\frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(model, config)

classLabels = []
file_name = cwd + r'\labels.txt'
with open(file_name, 'rt') as fpt:
  classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)
model.setInputSize(320, 320) ## as per config file - can be changed
model.setInputScale(1.0/127.5) ## 255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5)) ## mobilenet
model.setInputSwapRB(True)

img = cv2.imread('man_merc.jpg')
plt.imshow(img) ## BGR format
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) ## RGB format


ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
print(ClassIndex)

font_scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):   # flatten gets rid of nested arrays
  #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
  #cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontscale=font_scale, color=(0,0,0), thickness=1)
  cv2.rectangle(img, boxes, (255, 0, 0), 2)                                           # blue rectangles
  cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=2 )

  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
cap = cv2.VideoCapture(cwd + r"\source.mp4")
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open video")
    
while True:
      
      ret, frame = cap.read()
      ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)
    
      print(ClassIndex)
      if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
          if (ClassInd<=100):
              cv2.rectangle(frame, boxes, (255, 0, 0), 2)
              cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=2 )
    
      cv2.imshow('Object Detection',frame)
      out.write(frame)  
      if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



plt.show()
