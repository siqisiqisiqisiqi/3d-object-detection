#!/usr/bin/env python3
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if f"{parent}/src" not in sys.path:
    sys.path.insert(1, parent + "/src")

import rospy
import cv2

from object_detection.msg import BBox3d, Detection
from sensor_msgs.msg import Image

class DetectionNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        rospy.on_shutdown(self.shutdown)

        self.img = None
        # Init the detection node
        self.detection_msg = Detection()
        self.bbox_pub = rospy.Publisher('detections', Detection, queue_size=10)
        self.img_sub = rospy.Subscriber('zed2i/zed_node/left_raw/image_raw_color', Image, self.img_callback)
        # self.rate = rospy.Rate(int(rospy.get_param("~publish_rate")))
        self.rate = rospy.Rate(10)

    def img_callback(self, image):
        self.img = image
        cv2.imshow('Left Image', self.img)

    # def detection(self):



    def publish_bbox(self):
        for box in range(6):
            bbox_msg = BBox3d()
            bbox_msg.x1 = box
            bbox_msg.y1 = box
            bbox_msg.x2 = box
            bbox_msg.y2 = box
            bbox_msg.width = box
            bbox_msg.length = box
            self.detection_msg.header.seq = 10
            self.detection_msg.header.frame_id = 'odom'
            self.detection_msg.header.stamp = rospy.get_rostime()
            self.detection_msg.bbox_3d.append(bbox_msg)
        while not rospy.is_shutdown():
            self.bbox_pub.publish(self.detection_msg)
            self.rate.sleep()

    def shutdown(self):
        rospy.loginfo('shutting down!')


if __name__ == '__main__':
    try:
        node = DetectionNode('BBox3d')
        node.publish_bbox()
    except rospy.ROSInterruptException:
        pass
