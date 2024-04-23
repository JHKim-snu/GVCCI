import rospy
import sys
import cv2
import numpy as np
import random
import time
import darknet.darknet as darknet
import ros_numpy
import tf

from tf2_sensor_msgs import do_transform_cloud
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError

class Camera:

    def __init__(self, use_yolo=False):

        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber('/camera/color/image_rect_color', Image, self._rgb_cb)
        self.pc_sub  = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self._pc_cb)
        self.listener = tf.TransformListener()
        self.rgb_img = None
        self.cloud   = None
        self.use_yolo = use_yolo
        if use_yolo:
            random.seed(3)
            net, cls_names, cls_colors = darknet.load_network('./darknet/cfg/yolov4.cfg',
                                                              './darknet/cfg/coco.data',
                                                              './darknet/weights/yolov4.weights')
            self.net = net
            self.cls_names = cls_names
            self.cls_colors = cls_colors
            self.width = darknet.network_width(self.net)
            self.height = darknet.network_height(self.net)


    def _rgb_cb(self, data):

        try:
            cv_img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self.rgb_img = cv_img
        except CvBridgeError as e:
            rospy.logwarn('RGB Image Get Error: ' + str(e))
        return


    def _pc_cb(self, cloud):

        self.cloud = cloud
        return


    def get_rgb(self, timeout=1):

        start_time = time.time()
        while self.rgb_img is None:
            interval = time.time() - start_time
            if interval > timeout :
                rospy.logwarn("get_rgb timeout")
                return None
            rospy.sleep(0.1)
            continue
        return self.rgb_img


    def get_detections(self):

        if not self.use_yolo:
            rospy.logwarn('YOLO is not initialized')
            return None
        st = time.time()
        while self.rgb_img is None:
            interval = time.time() - st
            if interval > timeout :
                rospy.logwarn("get_detections timeout")
                return None
            rospy.sleep(0.1)
        detections = None
        while detections is None:
            cv_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB)
            rospy.loginfo(self.width)
            rospy.loginfo(self.height)
            rgb_img = cv2.resize(cv_img,
                                 (self.width, self.height),
                                 interpolation=cv2.INTER_LINEAR)
            darknet_img = darknet.make_image(self.width, self.height, 3)
            darknet.copy_image_from_bytes(darknet_img, cv_img.tobytes())
            detections = darknet.detect_image(self.net, self.cls_names, darknet_img)
            darknet.free_image(darknet_img)
        return detections


    def get_cloud(self, timeout=1):

        start_time = time.time()
        while self.cloud is None:
            interval = time.time() - start_time
            if interval > timeout :
                rospy.logwarn("get_cloud timeout")
                return None
            rospy.sleep(0.1)
            continue
        # Do transformation to base_link when get method called
        while True:
            try:
                trans, rot = self.listener.lookupTransform('base_link',
                                                           'camera_color_optical_frame',
                                                           rospy.Time(0))
                t = TransformStamped()
                t.transform.translation.x = trans[0]
                t.transform.translation.y = trans[1]
                t.transform.translation.z = trans[2]
                t.transform.rotation.x = rot[0]
                t.transform.rotation.y = rot[1]
                t.transform.rotation.z = rot[2]
                t.transform.rotation.w = rot[3]
                cloud_transformed = do_transform_cloud(self.cloud, t)
                cloud_transformed = ros_numpy.numpify(cloud_transformed)
                cloud_transformed = cloud_transformed.reshape(640, 480)
                #cloud_transformed = cloud_transformed.reshape(1280, 720)
                #cloud_transformed = cv2.resize(cloud_transformed, (640, 480))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn('cloud transformation exception')
                continue
        return cloud_transformed


if __name__ == '__main__':

    rospy.init_node('camera_test', disable_signals=True)
    YOLO = False
    camera = Camera(YOLO)

    #agent.arm.to_camera_pose()
    idx = 0
    while not rospy.is_shutdown():
        raw_input('Press any key to take a picture')
        rgb = camera.get_rgb()
        name = ('{}'.format(idx)).zfill(4)
        cv2.imwrite('samples/corl23/{}.png'.format(name), rgb)
        idx += 1
        print('image {} taken'.format(name))
        #val = raw_input('Continue? [y/n]')
        #if val != 'y': break
    '''
    print(np.nanmean(cloud['x'].shape))
    print(np.nanmean(cloud['y'].shape))
    print(np.nanmean(cloud['z'].shape))
    detection = camera.get_detections()
    print(detection)
    '''
    





