# Stereo-VO
Stereo Visual Odometry on EuRoC-MAV Dataset

## Workflow
This project presents a complete pipeline for stereo-vision based Odometry. The work is implemented in ROS2 Jazzy & Ubuntu 24.04. The methodology for the project is as follows:

**Image Rectification:** Epipolar rectification across stereo image pairs for epipolar lines to be horizontal and aligned with axes.  

**Depth Estimation:** Disparity & Depth Estimation to recover true metric scale and track camera motion.  

**Feature Detection & Matching:** Keypoint Detection and Matching handled using ORB + FLANN Matcher.  

**Motion Estimation:** Align tracked 3D landmarks to 2D projections using PnP + RANSAC.  

**Publishing the Results:** Publish the pose as a ROS2 message type in RViz, and plot it in real time.

## Results


https://github.com/user-attachments/assets/61e73e50-08c5-4002-ab2c-635f09ce377a


