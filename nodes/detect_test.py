#!/usr/bin/env python3
import threading
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
parent = os.path.dirname(root)
sys.path.insert(1, f'{root}/src/yolov5')
sys.path.insert(1, f'{root}/src')
sys.path.insert(1, root)

import rospy
import torch
import numpy as np
import pyzed.sl as sl
from std_msgs.msg import String
from collections import namedtuple
import torch.backends.cudnn as cudnn
from src.yolov5.models.experimental import attempt_load
from src.yolov5.utils.general import check_img_size
from src.yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from src.yolov5.utils.torch_utils import select_device
from src.yolov5.utils.augmentations import letterbox

from object_detection.msg import BBox3d, Detection


param = namedtuple("param","weights, svo, img_size, conf_thres")
lock = threading.Lock()


class CustomThread(threading.Thread):
    def __init__(self, opt):
        threading.Thread.__init__(self)
        self.weights = opt.weights
        self.img_size = opt.img_size
        self.conf_thres = opt.conf_thres
        self.iou_thres = 0.45
        self.image_net = None
        self.exit_signal = False
        self.run_signal = False
        self.detections = None

    def update_image(self, data):
        self.image_net = data

    def img_preprocess(self, device, half, net_size):
        net_image, ratio, pad = letterbox(self.image_net[:, :, :3], net_size, auto=False)
        net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        net_image = np.ascontiguousarray(net_image)

        img = torch.from_numpy(net_image).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, ratio, pad
    
    def xywh2abcd(self, xywh, im_shape):
        output = np.zeros((4, 2))

        # Center / Width / Height -> BBox corners coordinates
        x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
        x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
        y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
        y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

        # A ------ B
        # | Object |
        # D ------ C

        output[0][0] = x_min
        output[0][1] = y_min

        output[1][0] = x_max
        output[1][1] = y_min

        output[2][0] = x_min
        output[2][1] = y_max

        output[3][0] = x_max
        output[3][1] = y_max
        return output
    
    def detections_to_custom_box(self, detections, im):
        output = []
        for i, det in enumerate(detections):
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], self.image_net.shape).round()
                gn = torch.tensor(self.image_net.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    # Creating ingestable objects for the ZED SDK
                    obj = sl.CustomBoxObjectData()
                    shape = self.image_net.shape
                    # rospy.loginfo(f"shape is {shape}")
                    obj.bounding_box_2d = self.xywh2abcd(xywh, shape)
                    obj.label = cls
                    obj.probability = conf
                    obj.is_grounded = False
                    output.append(obj)
        return output

    def run(self):
        rospy.loginfo("Intializing Network...")

        device = select_device()
        half = device.type != 'cpu'  # half precision only supported on CUDA
        imgsz = self.img_size

        # Load model
        model = attempt_load(self.weights, device=device)  # load FP32
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16
        cudnn.benchmark = True

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        while not self.exit_signal:
            if self.run_signal:
                lock.acquire()
                img, ratio, pad = self.img_preprocess(device, half, imgsz)

                pred = model(img)
                pred = pred[0]
                det = non_max_suppression(pred, self.conf_thres, self.iou_thres)

                # ZED CustomBox format (with inverse letterboxing tf applied)
                self.detections = self.detections_to_custom_box(det, img)
                lock.release()
                self.run_signal = False
            rospy.sleep(0.01)


class DetectionNode:
    def __init__(self):

        rospy.init_node('bounding_box', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        self.sim = rospy.get_param("~sim")
        if self.sim == True:
            self.opt = param(f'{root}/src/weights/best2.pt', f'{parent}/svo_data/test1.svo', 416, 0.2)
        else:
            self.opt = param(f'{root}/src/weights/best2.pt', None, 416, 0.2)


        # start the keyboard detection
        self.capture_thread = CustomThread(self.opt)
        self.capture_thread.start()

        # Init the detection node
        self.detection_msg = Detection()
        
        self.bbox_pub = rospy.Publisher('obstacle', Detection, queue_size=10)
        self.sim_pub = rospy.Publisher('status', String, queue_size = 10)
        self.rate = rospy.Rate(10)

    def detect_obstacle(self):
        # Init the camera
        rospy.loginfo(f"Initializing Camera...")
        zed = sl.Camera()
        input_type = sl.InputType()
        if self.opt.svo is not None:
            input_type.set_from_svo_file(self.opt.svo)
            # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
        init_params.depth_maximum_distance = 50

        runtime_params = sl.RuntimeParameters()
        runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
        status = zed.open(init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            rospy.loginfo(repr(status))

        image_left_tmp = sl.Mat()
        rospy.loginfo(f"Initialized Camera")
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
        zed.enable_positional_tracking(positional_tracking_parameters)

        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = False
        zed.enable_object_detection(obj_param)

        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

        detection = Detection()
        detection.header.frame_id = 'base_link'
        detection.header.seq = 10

        rospy.loginfo("finish the initialization")
        while not rospy.is_shutdown():
            while not self.capture_thread.exit_signal:
                if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                    bbox = BBox3d()
                    detection.bbox_3d = []
                    detection.header.stamp = rospy.get_rostime()
                    # -- Get the image
                    lock.acquire()
                    zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                    image_net = image_left_tmp.get_data()
                    self.capture_thread.update_image(image_net)
                    lock.release()
                    self.capture_thread.run_signal = True
                    # -- Detection running on the other thread
                    while self.capture_thread.run_signal:
                        rospy.sleep(0.001)
                    # Wait for detections
                    lock.acquire()
                    # -- Ingest detections
                    zed.ingest_custom_box_objects(self.capture_thread.detections)
                    lock.release()
                    zed.retrieve_objects(objects, obj_runtime_param)
                    # Retrieve the object information
                    for object in objects.object_list:
                        if len(object.bounding_box) > 0:
                            bbox = BBox3d()
                            point_a = object.bounding_box[4, :]
                            point_b = object.bounding_box[7, :]
                            dx = point_b[0] - point_a[0]
                            dy = point_b[2] - point_a[2]
                            yaw = np.arctan2(dy, dx)
                            bbox.x_center, bbox.y_center = (point_a[0]+point_b[0])/2, (point_a[1]+point_b[1])/2
                            bbox.width, bbox.length = object.dimensions[0], object.dimensions[2]
                            bbox.yaw = yaw
                            detection.bbox_3d.append(bbox)
                        else:
                            pass
                    # rospy.loginfo(f"object list is {object_list}")
                    rospy.loginfo("finish the object detection")
                    self.bbox_pub.publish(detection)
                    self.sim_pub.publish('Done')
                    # rospy.loginfo("successfully publish the detection info")
                    self.rate.sleep()

                else:
                    self.capture_thread.exit_signal = True

    def shutdown(self):
        self.capture_thread.exit_signal = True
        rospy.loginfo('shutting down!')


if __name__ == '__main__':

    try:
        node = DetectionNode()
        node.detect_obstacle()
    except rospy.ROSInterruptException:
        pass