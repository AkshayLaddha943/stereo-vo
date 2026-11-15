import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import numpy as np
import cv2

left_cam_topic = "/cam0/image_raw"
right_cam_topic = "/cam1/image_raw"

#cam0 intrinsic
K0 = np.array([[458.654, 0, 367.215],
              [0, 457.296, 248.375],
              [0, 0, 1]], dtype=np.float64)

D0 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float64)

K1 = np.array([[457.587, 0, 379.999],
               [0, 456.134, 255.238],
               [0, 0, 1]], dtype=np.float64)

D1 = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05], dtype=np.float64)

T0 = np.array([
            [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
            [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
            [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
            [0, 0, 0, 1]], dtype=np.float64)

T1 = np.array([
            [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
            [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
            [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
            [0, 0, 0, 1]], dtype=np.float64)

T_10 = np.linalg.inv(T1) @ T0

R = T_10[:3, :3]
T = T_10[:3, 3]

class GetDepth(Node):
    def __init__(self, ):
        super().__init__('get_depth')

        self.image_size = (752, 480)
        self.bridge = CvBridge()

        R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(K0, D0, K1, D1, self.image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, 
                                                    alpha=0)

        #rectification maps
        self.map0_x, self.map0_y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, self.image_size,cv2.CV_32FC1)
        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, self.image_size, cv2.CV_32FC1)

        self.focal_length = P0[0,0]
        self.baseline = 0.1100

        self.img1_sub = Subscriber(self, Image, left_cam_topic, qos_profile_sensor_data)
        self.img2_sub = Subscriber(self, Image, right_cam_topic, qos_profile_sensor_data)

        self.time_sync = ApproximateTimeSynchronizer([self.img1_sub, self.img2_sub], queue_size=10, slop=0.02)
        self.time_sync.registerCallback(self.right_callback)

        self.depth_pub = self.create_publisher(Image, "/depth/image", qos_profile_sensor_data)
        self.depth_norm_pub = self.create_publisher(Image, "/norm_depth/image", qos_profile_sensor_data)

        self.get_logger().info("Depth Node Initialized")

    def right_callback(self, leftmsg: Image, rightmsg: Image):
        left_img = self.bridge.imgmsg_to_cv2(leftmsg, desired_encoding="mono8")
        right_img = self.bridge.imgmsg_to_cv2(rightmsg, desired_encoding="mono8")

        left_rect = cv2.remap(left_img, self.map0_x, self.map0_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map1_x, self.map1_y, cv2.INTER_LINEAR)

        sgbm = cv2.StereoSGBM_create(numDisparities=128, blockSize=7)
        disparity = sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0

        depth_image = np.where(disparity > 0, (self.focal_length * self.baseline) / disparity, 0)
        depth_image = cv2.medianBlur(depth_image, 5)
        norm_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_image, "32FC1"))
        self.depth_norm_pub.publish(self.bridge.cv2_to_imgmsg(norm_depth, "mono8"))


def main(args=None):
    rclpy.init(args=args)
    node = GetDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()