#import sys
#import time
import cv2
import matplotlib as mpl
mpl.use('GTKAgg')
from matplotlib import pyplot as plt
#import numpy as np
 
import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
 
class RectBuilder:
    def __init__(self, rect, x, y, width, height):
        self.rect = rect
        self.x, self.y, self.width, self.height = x, y, width, height
       
        self.first_click = True       
        self.rect.figure.canvas.mpl_connect('button_press_event', self._click)
       
#        self.label = ''
#        self.rect.figure.canvas.mpl_connect('key_press_event', self._press)
       
    def _click(self, event):
        logger.debug('Event: %s', event)
 
        if event.inaxes != self.rect.axes: return
        logger.debug('Click at (%d, %d)', event.xdata, event.ydata)
 
        if self.first_click:
            self.x = event.xdata
            self.rect.set_x(self.x)
            self.y = event.ydata
            self.rect.set_y(self.y)
            self.first_click = False
        else:
            self.width = event.xdata - self.x
            self.rect.set_width(self.width)
            self.height = event.ydata - self.y
            self.rect.set_height(self.height)
            self.first_click = True
 
        logger.info('(%d, %d), (%d, %d)',
                    self.x, self.y,
                    self.x + self.width, self.y + self.height)
        self.rect.figure.canvas.draw()
   
#    def _press(self, event):
#        logger.debug('Event: %s', event)
#        logger.debug('Press %c', event.key)
#        if event.key == "~":
#            self.label = self.label[:-1]
#        else:
#            self.label += event.key
#        self.rect.set_label(self.label)
       
cap = cv2.VideoCapture(0)
 
# Capture a frame
ret, frame = cap.read()
#cap.release()
b,g,r = cv2.split(frame)
image = cv2.merge([r,g,b])
 
# Plot image

mpl.rcParams['toolbar'] = 'None'
fig = plt.figure()
ax_img = fig.add_subplot(111)
ax_img.imshow(image)
 
# Plot bounding box and wait for events
ax_bb = fig.add_subplot(111)
ax_bb.set_title('Click to choose bounding box:')
x, y, width, height = 10, 10, 50, 50
rect = ax_bb.add_patch(mpl.patches.Rectangle((x, y),
                                             width,
                                             height,
                                             fill = False))
rect_builder = RectBuilder(rect, x, y, width, height)
plt.show()
x1, y1 = int(rect.get_x()), int(rect.get_y())
x2, y2 = int(x1 + rect.get_width()), int(y1 + rect.get_height())
 
#cropped_image = image[y1:y2, x1:x2]
#fig = plt.figure()
#ax_img = fig.add_subplot(111)
#ax_img.imshow(cropped_image)
#plt.show()
 
#cap = cv2.VideoCapture(0)
while(True):
    # Read webcam frames and decode
    ret, frame = cap.read()
#    b,g,r = cv2.split(frame)
#    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_cropped = image_gray[y1:y2, x1:x2]
    #image_scaled = cv2.resize(image_cropped, )
    cv2.imshow('frame', image_cropped)
 
    # Scan QRcode from image_gray
 
#    # Save image and QRCode data
#    img_fname = str(int(time.time())) + ".png"
#    cv2.imwrite(img_fname, image_cropped)
#    time.sleep(2)
#    k = cv2.waitKey(30) & 0xff
    k = cv2.waitKey()
    if k == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()