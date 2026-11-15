from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from message_filters import Subscriber, ApproximateTimeSynchronizer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


left_cam_topic = "/cam0/image_raw"
right_cam_topic = "/cam1/image_raw"

FLANN_INDEX_LSH = 6
index_param = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_param = dict(checks=50)

class StereoVisualOdometry(Node):
    def __init__(self, ):
        super().__init__('svo')

        self.bridge = CvBridge()
        self.svo_path = Path()
        self.svo_path.header.frame_id = "vo_world"

        self.camera_intrinsics()
        self.rectify()

        self.img1_sub = Subscriber(self, Image, left_cam_topic, qos_profile_sensor_data)
        self.img2_sub = Subscriber(self, Image, right_cam_topic, qos_profile_sensor_data)

        #synchronize
        self.time_sync = ApproximateTimeSynchronizer([self.img1_sub, self.img2_sub], queue_size=10, slop=0.02)
        self.time_sync.registerCallback(self.odom_callback)

        self.svo_pub = self.create_publisher(Path, "/vo/path", 10)

        #feat detect & matcher
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.matcher = cv2.FlannBasedMatcher(index_param, search_param)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        #states
        self.pose = np.eye(4, dtype=np.float64)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.prev_depth = None

        self.svo_traj = []
        self.frame_count = 0

        #3d plot
        plt.ion()
        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_title("SVO")
        self.ax.set_xlabel("X displacement")
        self.ax.set_ylabel("Y displacement")
        self.ax.set_zlabel("Z displacement")

        self.ax.grid(True, linestyle="--", alpha=0.5)

        self.vo_line, = self.ax.plot([], [], [], color="orange", linewidth=1.5, label="VO Trajectory")
        self.vo_start, = self.ax.plot([], [], [], "bo", markersize=6, label="Start")
        self.vo_current, = self.ax.plot([], [], [], "ro", markersize=5)

        self.ax.legend()
        plt.show(block=False)


    def camera_intrinsics(self):
        self.cam0_intrinsic = np.array([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1]], 
                                       dtype=np.float64)

        self.cam1_intrinsic = np.array([[457.587, 0, 379.999], [0, 456.134, 255.238], [0, 0, 1]], 
                                       dtype=np.float64)
        self.cam0_distortion = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float64)
        self.cam1_distortion = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05], dtype=np.float64)

        self.T_cam0 = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975], 
                                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                                [0, 0, 0, 1]], dtype=np.float64)
        
        self.T_cam1 = T1 = np.array([[0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556], 
                                     [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
                                     [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
                                     [0, 0, 0, 1]], dtype=np.float64)

    def rectify(self):
        T_10 = np.linalg.inv(self.T_cam1) @ self.T_cam0

        R = T_10[:3, :3]
        T = T_10[:3, 3]

        image_size = (752, 480)

        R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(self.cam0_intrinsic, self.cam0_distortion, self.cam1_intrinsic, self.cam1_distortion, 
                                                    image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        self.map0_x, self.map0_y = cv2.initUndistortRectifyMap(self.cam0_intrinsic, self.cam0_distortion, R0, P0, image_size,cv2.CV_32FC1)
        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(self.cam1_intrinsic, self.cam1_distortion, R1, P1, image_size, cv2.CV_32FC1)

        self.K = P0[:3, :3]
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        self.baseline = abs(P1[0, 3] / P1[0, 0])

    def preprocess(self, msg, left=True):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")    

        if left:
            rect = cv2.remap(img, self.map0_x, self.map0_y, cv2.INTER_LINEAR)
        else:
            rect = cv2.remap(img, self.map1_x, self.map1_y, cv2.INTER_LINEAR)

        #clahe
        enhanced_img = self.clahe.apply(rect)

        return enhanced_img

    def compute_disparity(self, left_rectified, right_rectified):
        stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=7)
        disparity = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
        return disparity

    def compute_depth(self, disparity, baseline, fx):
        depth = (fx * baseline) / (disparity + 1e-5)
        depth[disparity <= 0] = 0
        return depth

    def odom_callback(self, left_msg, right_msg):
        left_img = self.preprocess(left_msg, left=True)
        right_img = self.preprocess(right_msg, left=False)

        #compute depth
        disparity = self.compute_disparity(left_img, right_img)
        depth = self.compute_depth(disparity, self.baseline, self.fx)

        kp_left, desc_left = self.orb.detectAndCompute(left_img, None)

        if desc_left is None:
            self.get_logger().debug("Skipping frame with missing ORB descriptors")
            return

        self.frame_count += 1
        if self.prev_desc is None:
            self.prev_kp = kp_left
            self.prev_desc = desc_left
            self.prev_depth = depth
            self.publish_path(left_msg.header.stamp)
            self.plot()
            return

        matches = self.matcher.knnMatch(self.prev_desc, desc_left, k=2)

        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < float(0.7) * n.distance:
                good_matches.append(m)

        success = self.compute_odometry(good_matches, kp_left)

        if success:
            self.publish_path(left_msg.header.stamp)
            self.plot()

        self.prev_kp = kp_left
        self.prev_desc = desc_left
        self.prev_depth = depth

    def compute_odometry(self, matches, curr_kp):
        img_pts = []
        obj_pts = []

        height, width = self.prev_depth.shape

        for m in matches:
            u, v = self.prev_kp[m.queryIdx].pt
            x = int(round(u))
            y = int(round(v))

            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            Z = self.prev_depth[y, x]

            if not np.isfinite(Z) or Z <= 0 or Z > 30:
                continue

            # Reconstruct 3D point in previous camera frame
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy

            obj_pts.append([X, Y, Z])
            img_pts.append(curr_kp[m.trainIdx].pt)

        if len(obj_pts) < 10:
            return False

        object_pts = np.array(obj_pts, dtype=np.float32)
        image_pts = np.array(img_pts, dtype=np.float32)

        success, R, t, inlier = cv2.solvePnPRansac(object_pts, image_pts, self.K, None,
                                                   iterationsCount=100, reprojectionError=3.0, confidence=0.99,
                                                   flags=cv2.SOLVEPNP_EPNP)


        if not success:
            return False

        inlier_count = len(inlier)
        inlier_ratio = inlier_count / len(object_pts)

        if inlier_count < 20 or inlier_ratio < 0.5:
            return False

        if not success or inlier is None:
            return False
        
        rot_mat, _ = cv2.Rodrigues(R)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot_mat
        T[:3, 3] = t.flatten()

        self.pose = self.pose @ np.linalg.inv(T)
        position = self.pose[:3, :3]

        return True


    def plot(self):
        if len(self.svo_traj) == 0:
            return

        points = np.array(self.svo_traj)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        self.vo_line.set_data(x,y)
        self.vo_line.set_3d_properties(z)

        self.vo_start.set_data([x[0]], [y[0]])
        self.vo_start.set_3d_properties([z[0]])

        self.vo_current.set_data([x[-1]], [y[-1]])
        self.vo_current.set_3d_properties([z[-1]])

        center = (points.max(axis=0) + points.min(axis=0)) / 2
        radius = max(np.ptp(points, axis=0).max() / 2, 1.0)

        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def publish_path(self, stamp):
        position = self.pose[:3, 3]
        quaternion = R.from_matrix(self.pose[:3, :3]).as_quat()

        pose_msg = PoseStamped()

        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "vo_world"

        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])

        self.svo_path.header.stamp = stamp
        self.svo_path.poses.append(pose_msg)

        self.svo_pub.publish(self.svo_path)

        self.svo_traj.append(position.copy())


def main(args=None):
    rclpy.init(args=args)
    node = StereoVisualOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        plt.close('all')
        rclpy.shutdown()

if __name__ == "__main__":
    main()