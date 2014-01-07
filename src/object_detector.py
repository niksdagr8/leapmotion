#!/usr/bin/env python

# Author: Nikhil Soni
# Date: December 3, 2013
# This ROS node is designed to take the live camera feed from Baxter's wrist
# and recognize objects placed on a table by matching the shape and mean HSV
# value in the object's contour. The node reads the trained shape and HSV value 
# from a bagfile, created by the Training node.

import roslib
from roslaunch.loader import rosparam
from rospy_tutorials.msg._Floats import Floats
#import baxter_msgs
roslib.load_manifest('table_detector')
import sys
import rospy
import cv
import cv2
import numpy
import baxter_interface
import rosparam
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
from std_msgs.msg import Int32
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from copy import deepcopy
import rosbag

class table_detector:
"""
Class for detecting objects placed on a table using Baxter's wrist camera
"""
  def __init__(self):
      
        
    limb = 'left'
    cv.NamedWindow("Image window0", 1)
    cv.NamedWindow("Image window1", 1)
    cv.NamedWindow("Image window2", 1)
    cv.NamedWindow("Image window3", 1)
    cv.NamedWindow("Image window4", 1)
    cv.NamedWindow("Image window5", 1)
    
    self.bridge = CvBridge()
    self.head_pub = rospy.Publisher('/sdk/xdisplay', Image, latch=True)
    
    ##OPEN BAG FILE FOR TASKS
    task_num = rospy.get_param("test_task_number")
    pre_path = "/opt/ros/groovy/stacks/baxterjd/tasks/left/"
    file_name = "kitting_task"+str(task_num)+".bag"
    bagfile = rosbag.Bag(pre_path+file_name,'r')
    # Read required information
    for topic, msg, time in bagfile.read_messages():
        if topic == 'train_hsv_val':
            self.hsv_callback(msg)
        elif topic == 'train_mean_eigen_list':
            self.mean_eigen_callback(msg)
        elif topic == 'train_shape_topic':
            self.shape_callback(msg)
            
    bagfile.close()        
   
    
    #Publishes the center pixel coordinate of the detected object
    self.mean_pub = rospy.Publisher('detected_object_error', numpy_msg(Floats))
    
    
    self.image_sub = rospy.Subscriber("/cameras/"+limb+"_hand_camera/image",Image,self.callback)
    self.hessian_threshold = 10
    self.image = Image()
    
    
    self.fingers_calibrated = False
    
    self.gripper = baxter_interface.Gripper(limb)
    
    open = numpy.asarray(cv2.cv.LoadImage("/opt/ros/groovy/stacks/baxterjd/table_detector/src/calibration_open.jpg")[:,:])
    close = numpy.asarray(cv2.cv.LoadImage("/opt/ros/groovy/stacks/baxterjd/table_detector/src/calibration_close.jpg")[:,:])
    
    self.calibration_open = cv2.cvtColor(open,cv2.COLOR_BGR2GRAY)
    self.calibration_close = cv2.cvtColor(close,cv2.COLOR_BGR2GRAY)
    
    self.subscribed_topic = False
    
  def wheel_changed(self,val):
      self.hessian_threshold = 10*val
      
      if self.hessian_threshold < 10:
          self.hessian_threshold = 10
      
  def button_changed(self,val):
      if val == True:
          self.train_image_pub.publish(self.image)
          a = numpy.array([[1.3,4.2,3.5,4.6,5.7],[2.2,1.2,3.4,1.1,1.1]],dtype=numpy.float32)
          self.shape_pub.publish(numpy.array(self.shape.reshape(self.shape.shape[1]*2),dtype=numpy.float32))
  
  def mean_eigen_callback(self,data):
      arr = numpy.array(data.data)
      self.train_mean_eigen = arr.reshape((arr.shape[0]/4,4))
      
  def hsv_callback(self,data):
      arr = numpy.array(data.data)
      self.hsv_val = arr.reshape((arr.shape[0]/3,3))
      
  def shape_callback(self,data):
      
      marker = 99
      ar = numpy.array(data.data)
      
      arr = ar.tolist()
      
      length = len(arr)-1
      self.train_template_list = []
      self.train_template_sum_list = []
      #index = arr.index(marker)
      last = False
      
      
      #Parse incoming array
      while last == False: 
          index = arr.index(marker)
          print "index",index
          list = arr[0:index]
          arr = arr[index+1::]
          
          list_arr = numpy.array(list,dtype=numpy.float32)
          
          #Reshape to get back 2xN point set
          list_arr = list_arr.reshape((2,list_arr.shape[0]/2))
          
          
          if index >= len(arr)-1:
              last = True
              
          template_list = []    
          
          #Normal upright
          self.train_template_1 = self.shape_to_template(list_arr)
          template_list.append(self.train_template_1)
      
          #Vertically inverted
          self.train_template_2 = cv2.flip(self.train_template_1,1)
          template_list.append(self.train_template_2)
      
          #Upright horizontally inverted
          self.train_template_3 = cv2.flip(self.train_template_1,0)
          template_list.append(self.train_template_3)
      
          #Vertically inverted and Horizontally inverted
          self.train_template_4 = cv2.flip(self.train_template_1,-1)
          template_list.append(self.train_template_4)
          
          #Append list of 4 templates to train_template_list
          self.train_template_list.append(template_list)
          
          self.train_template_sum_list.append(numpy.sum(self.train_template_1))
      self.subscribed_topic = True
      
  def threshold_image(self,image):
  """
  Function to carry out adadptive thresholding, converting the input image to a binary image
  """
      hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
      im_blur = cv2.GaussianBlur(hsv_image,(9,9),11)
    
      hue_mean = numpy.mean(hsv_image[:,:,0])
      sat_mean = numpy.mean(hsv_image[:,:,1])
      val_mean = numpy.mean(hsv_image[:,:,2])
    
    
      hue_std = numpy.std(hsv_image[:,:,0])
      sat_std = numpy.std(hsv_image[:,:,1])
      val_std = numpy.std(hsv_image[:,:,2])
    
      div = 1.8
      img_1 = cv2.inRange(im_blur,(hue_mean-div*hue_std,sat_mean-div*sat_std,val_mean-div*val_std),(hue_mean+div*hue_std,sat_mean+div*sat_std,val_mean+div*val_std))
      
      img = cv2.inRange(img_1,0,0.5)
      return img
    
  def filter_contours(self,contours,heirarchy,thresh_area):
  """
  Filter contours based on heirarchy and area
  """
    filtered = []
    
    counter = 0
    for c in contours:   
        if cv2.contourArea(c,False) > thresh_area:
            #print cv2.contourArea(c,False)
            if heirarchy[0][counter][3] == -1:
                filtered.append(c)
        counter = counter + 1    
    return filtered

  def angle_with_x(self,b):
      if b[0] >= 0 and b[1] >=0:
          theta = numpy.arccos(b[0])
      elif b[0] < 0 and b[1] >= 0:
          theta = (-1*numpy.arccos(abs(b[0])) + numpy.pi)
      elif b[0] < 0 and b[1] < 0:
          theta = -1*(-1*numpy.arccos(abs(b[0]))+numpy.pi)
      elif b[0] >= 0 and b[1] < 0:
          theta = -1*(numpy.arccos(b[0]))
      
      return theta
  
  def contour_PCA(self,contours):
  """
  Returns the Eigen vectors, Eigen values and meanof a set of contour points, similar to Principal Component Analysis
  """
      contour = []
      for c in contours:
          contour.append(c[0])
      
      points = numpy.array(contour,dtype=numpy.float32).T
      
      
      #Find covariance of points
      cov = numpy.cov(points)
      
      #Find eigen vectors and values
      eig_val, eig_vec = numpy.linalg.eig(cov)
      
      #Transform points to Eigen vector frame
      if eig_val[0] > eig_val[1]: #find transformation matrix
          Trans = numpy.array([[eig_vec[0][0],eig_vec[1][0]],[eig_vec[0][1],eig_vec[1][1]]])
      else:
          Trans = numpy.array([[eig_vec[1][0],eig_vec[0][0]],[eig_vec[1][1],eig_vec[0][1]]])
      
      #Find mean
      mean = (numpy.mean(points[0]),numpy.mean(points[1]))
      
      
      points[0] = points[0] - numpy.mean(points[0])
      points[1] = points[1] -numpy.mean(points[1])
      
      
      #Transpose to get Nx2
      p = points.T
      
      
      b = numpy.array([Trans[0][0],Trans[1][0]])
      
      if b[0] >= 0 and b[1] >=0:
          theta = -1*numpy.arccos(b[0])
      elif b[0] < 0 and b[1] >= 0:
          theta = -1*(-1*numpy.arccos(abs(b[0])) + numpy.pi)
      elif b[0] < 0 and b[1] < 0:
          theta = -1*(numpy.arccos(abs(b[0]))+numpy.pi)
      elif b[0] >= 0 and b[1] < 0:
          theta = -1*(2*numpy.pi -1*numpy.arccos(b[0]))
      
      
          
      #print "theta",theta*180/numpy.pi
      
      R = numpy.array([[numpy.cos(theta),-1*numpy.sin(theta)],[numpy.sin(theta),numpy.cos(theta)]])
      #Apply transformation and go back to 2xN
      pts = numpy.dot(p,R).T
      
      norm_pts = self.normalize(pts)
      
      return eig_val, eig_vec, mean, norm_pts
  
  def normalize(self,pts):
      #Normalize points
      Scale0 = max(pts[0]) - min(pts[0])#numpy.linalg.norm(pts[0])
      Scale1 = max(pts[1]) - min(pts[1])
      
      if Scale0 > Scale1:
          pts = pts/Scale0
      else:
          pts = pts/Scale1
      
      return pts
      
  def shape_to_contour(self,shape):
      list = []
      for s in shape.T:
          list.append([s.tolist()])
      contour = numpy.array(list)
      
      return contour
  def shape_to_template(self,shape):
      #Converts set of 2D points to a numpy array image
      
      size = 300
      #Create template image
      template = numpy.zeros((size,size))
      for point in shape.T:
          x = int(point[0]*size*0.85) + size/2 #+ size/10
          y = int(point[1]*size*0.85) + size/2 #+ size/10
          template[y,x] = 255
          
      template = cv2.dilate(template,None,30,(-1,-1),3)
      return template
  
  def boundingrect(self,cont):
  """
  Calculate the bounding rectangle of a set of 2D contour points
  """
      contour = []
      for c in cont:
          contour.append(c[0])
      
      points = numpy.array(contour,dtype=numpy.int32).T
      minx = numpy.min(points[0])
      miny = numpy.min(points[1])
      maxx = numpy.max(points[0])
      maxy = numpy.max(points[1])
      
      return (minx,miny,maxx-minx,maxy-miny)
          
  def contour_hsv(self,image,mask,contour):
  """
  Calculate the mean HSV values for pixels inside a contour's bounding rectangle
  """
      roi = self.boundingrect(contour)
      
      img = image#cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
      count = 0
      hue =0 
      sat =0
      val =0
      
      for x in range(roi[0],roi[0]+roi[2]):
          for y in range(roi[1],roi[1]+roi[3]):
              if mask.item(y,x):
                  count = count + 1
                  hue = hue + img.item(y,x,0)
                  sat = sat + img.item(y,x,1)
                  val = val + img.item(y,x,2)
      
      return (hue/count,sat/count,val/count)
                  
  def callback(self,data):
    """
    Main callback function for image topic through Baxter's wrist camera
    """
      
    try:
      cv_image = self.bridge.imgmsg_to_cv(data, "bgr8")
    except CvBridgeError, e:
      print e
      
    self.image = data
    
    #convert to numpy array
    num_im = numpy.asarray(cv_image[:,:])
    num_im_copy = num_im
    
    #Threshold the image to convert to a binary image
    img_thresh = self.threshold_image(num_im)
    #img_temp = cv2.cv.fromarray(img_thresh.copy())
    
    #Show binary image
    cv.ShowImage("Image window2", cv2.cv.fromarray(img_thresh))
    
    #Apply finger masks
    if self.gripper.position() > 50:
        img = cv2.bitwise_and(img_thresh,self.calibration_open)
        #img = img_thresh * self.calibration_open  
    else:
        img = cv2.bitwise_and(img_thresh,self.calibration_close)
        #img = img_thresh * self.calibration_close
            
    
    #Erode and Dilate to remove noise and small blobs
    img = cv2.erode(img,None,30,(-1,-1),1)
    img = cv2.dilate(img,None,30,(-1,-1),1)
    mask = img
    
    #Find contours and show on original image
    contours, heirarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    #filter out small contours (less than threshold area)
    thresh_area = 300
    filtered_contours = self.filter_contours(contours,heirarchy, thresh_area)
    
    error_list = []
    dist_list = []
    
    # Start matching shape
    max_score = 0
    count = -1
    for fc in filtered_contours:
        
        count = count + 1 
        val,vec, mean, test_shape = self.contour_PCA(fc)
    
        test_template = self.shape_to_template(test_shape)
        test_sum  = numpy.sum(test_template)
        
        
        
        scores = []
        indices = []
        counter = 0
        
        for train_templates in self.train_template_list:
            scores_1 = []
            for train_template in train_templates:
                anded = cv2.bitwise_and(test_template,train_template)
                #cv.ShowImage("Image window5", cv2.cv.fromarray(anded))
                anded_sum = numpy.sum(anded)
                score = (anded_sum*anded_sum)/(self.train_template_sum_list[counter] * test_sum)
                scores_1.append(score)
            
            #cv2.namedWindow("win"+str(counter))
            #cv2.imshow("win"+str(counter),train_template)
        
            max_s = max(scores_1)
            ind = scores_1.index(max_s)
            
            scores.append(max_s)
            indices.append(ind)
            
            counter = counter + 1
            
        ms = max(scores)    
        #print "score",ms
        index = scores.index(ms)
        
        if ms > max_score:
            cv.ShowImage("Image window1", cv2.cv.fromarray(test_template))
            cv.ShowImage("Image window4", cv2.cv.fromarray(self.train_template_list[index][indices[index]]))
            anded = cv2.bitwise_and(test_template,self.train_template_list[index][indices[index]])
            cv.ShowImage("Image window3", cv2.cv.fromarray(anded))
            max_score = ms
            
        #cv2.line(num_im, (int(mean[0]),int(mean[1])), (int(mean[0]+50*vec[0][0]),int(mean[1]-50*vec[0][1])),(0,0,255),3)
        #cv2.line(num_im, (int(mean[0]),int(mean[1])), (int(mean[0]+50*vec[1][0]),int(mean[1]-50*vec[1][1])),(255,0,0),3)
        
        if ms > 0.10:
            cont = deepcopy(fc)
            hsv_val =  self.contour_hsv(num_im_copy, mask, cont)

            #hsv_dist = numpy.power(numpy.power(self.hsv_val[index][0]-hsv_val[0],2),0.5)
            hsv_dist = numpy.power(numpy.power(self.hsv_val[index][0]-hsv_val[0],2)+numpy.power(self.hsv_val[index][1]-hsv_val[1],2)+numpy.power(self.hsv_val[index][2]-hsv_val[2],2),0.5)
            #print self.hsv_val
            #print hsv_val
            #print "hsv_dist",hsv_dist
            #color = (0,255,0)
            #thickness = 4
            if hsv_dist < 60:
                color = (0,255,0)
                thickness = 4
                if val[0] > val[1]:
                    vec1 = vec[0][0]
                    vec2 = vec[0][1]
                else:
                    vec1 = vec[1][0]
                    vec2 = vec[1][1]
                    
                mean_arr = numpy.array((int(mean[0]) - numpy.shape(num_im)[1]/2, int(mean[1]) - numpy.shape(num_im)[0]/2, vec1, vec2),dtype=numpy.float32)
                theta_des = self.angle_with_x(self.train_mean_eigen[index][2::])
                theta_curr = self.angle_with_x(mean_arr[2::])
                
                error = numpy.array((self.train_mean_eigen[index][0] - mean_arr[0], self.train_mean_eigen[index][1] - mean_arr[1], (theta_des - theta_curr)*180/numpy.pi),dtype=numpy.float32)
                
                error_list.append(error)
                dist = numpy.linalg.norm(error[0:2])
                dist_list.append(dist)
                
            else:
                color = (255,0,0)
                thickness = 2
        else:
            color = (0,0,255)
            thickness = 2
        #Draw contours on original image
        cv2.drawContours(num_im,fc,-1,color,thickness)
        
    if len(error_list) > 0:
        index = dist_list.index(min(dist_list))
        error = error_list[index]
        #publish error
        self.mean_pub.publish(error)
    else:
        error = numpy.array([-1000.0,-1000.0,-1000.0],dtype=numpy.float32)
        self.mean_pub.publish(error)
            
    #print "=========="
    
    
    
    
    
    self.head_pub.publish(self.bridge.cv_to_imgmsg(cv2.cv.fromarray(num_im), "bgr8"))
    cv.ShowImage("Image window0", cv2.cv.fromarray(num_im))
    
    
    cv.WaitKey(1)

    
def main(args):
  
  rospy.init_node('table_detector_test', anonymous=True)
  ic = table_detector()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
