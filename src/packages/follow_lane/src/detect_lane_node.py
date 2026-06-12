#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from datetime import datetime
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage
from enum import Enum
import yaml
import util
from std_srvs.srv import SetBool, SetBoolResponse
# from ultralytics import YOLO   # disabled — using HSV white mask

#from duckietown.dtros import DTROS, NodeType

class DetectLaneNode:
    def __init__(self, node_name):
        # initialize the ROS node
        rospy.init_node(node_name)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name, self.cbUpdateParameters)

        # YOLO model disabled — white line detected via HSV thresholding
        # model_path = "/workspace/src/packages/follow_lane/src/model/best.pt"
        # self.yolo_model = YOLO(model_path)
        # self.class_white = 2  # white line class ID

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image_original = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbFindLane, queue_size=1)
        self.pub_lane = rospy.Publisher(f'/{self._vehicle_name}/detect/lane', Float64, queue_size=1)

        self._crop_im_size = 400
        self.is_running = False
        self.counter = 0

        self.video_writer = None
        self.video_out_path = "/workspace/src/videos/lane_detection_video.mp4"
        self.fps = 30
        self.video_record_enable = False
        self.display_image = None

        # Create service to toggle video recording
        self.srv_toggle_video = rospy.Service(f"/{self._vehicle_name}/toggle_video_recording", SetBool, self.cbToggleVideoRecording)

        # init debug channels
        self.pub_debug_lane   = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_croped/compressed', CompressedImage, queue_size=1)
        self.pub_debug_white  = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_white/compressed',  CompressedImage, queue_size=1)
        self.pub_debug_yellow = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_yellow/compressed', CompressedImage, queue_size=1)
        self.pub_red_line     = rospy.Publisher(f'/{self._vehicle_name}/detect/red_line', Float64, queue_size=1)
        self.pub_debug_red    = rospy.Publisher(f'/{self._vehicle_name}/debug/lane_red/compressed',    CompressedImage, queue_size=1)

    def cbUpdateParameters(self, parameters):
        # Update white line parameters
        self.hue_white_l        = parameters["white"]["hl"]["default"]
        self.hue_white_h        = parameters["white"]["hh"]["default"]
        self.saturation_white_l = parameters["white"]["sl"]["default"]
        self.saturation_white_h = parameters["white"]["sh"]["default"]
        self.lightness_white_l  = parameters["white"]["vl"]["default"]
        self.lightness_white_h  = parameters["white"]["vh"]["default"]

        # Update yellow line parameters
        self.hue_yellow_l        = parameters["yellow"]["hl"]["default"]
        self.hue_yellow_h        = parameters["yellow"]["hh"]["default"]
        self.saturation_yellow_l = parameters["yellow"]["sl"]["default"]
        self.saturation_yellow_h = parameters["yellow"]["sh"]["default"]
        self.lightness_yellow_l  = parameters["yellow"]["vl"]["default"]
        self.lightness_yellow_h  = parameters["yellow"]["vh"]["default"]

        # Red line parameters
        self.hue_red_l        = parameters["red"]["hl"]["default"]
        self.hue_red_h        = parameters["red"]["hh"]["default"]
        self.saturation_red_l = parameters["red"]["sl"]["default"]
        self.saturation_red_h = parameters["red"]["sh"]["default"]
        self.lightness_red_l  = parameters["red"]["vl"]["default"]
        self.lightness_red_h  = parameters["red"]["vh"]["default"]

        # Perspective transform points
        self.top_left_x     = parameters["crop_image"]["top_left_x"]["default"]
        self.top_left_y     = parameters["crop_image"]["top_left_y"]["default"]
        self.top_right_x    = parameters["crop_image"]["top_right_x"]["default"]
        self.top_right_y    = parameters["crop_image"]["top_right_y"]["default"]
        self.bottom_left_x  = parameters["crop_image"]["bottom_left_x"]["default"]
        self.bottom_left_y  = parameters["crop_image"]["bottom_left_y"]["default"]
        self.bottom_right_x = parameters["crop_image"]["bottom_right_x"]["default"]
        self.bottom_right_y = parameters["crop_image"]["bottom_right_y"]["default"]

    def cbToggleVideoRecording(self, req):
        """ROS service to toggle video recording on/off"""
        self.video_record_enable = req.data
        if req.data:
            return SetBoolResponse(success=True, message="Video recording enabled")
        else:
            if self.video_writer is not None and self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
            return SetBoolResponse(success=True, message="Video recording disabled")

    def get_red_pixel_count(self, mask):
        return int(np.count_nonzero(mask))

    def crop_img(self, img):
        img = img.copy()

        pts1 = np.float32([
            [self.top_left_x,     self.top_left_y],
            [self.top_right_x,    self.top_right_y],
            [self.bottom_right_x, self.bottom_right_y],
            [self.bottom_left_x,  self.bottom_left_y],
        ])

        pts2 = np.float32([
            [0,                  0],
            [self._crop_im_size, 0],
            [0,                  self._crop_im_size],
            [self._crop_im_size, self._crop_im_size],
        ])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (self._crop_im_size, self._crop_im_size))

    def get_x_for_driving(self, mask, distance, no_lane_value, left_line):
        grad = cv2.Sobel(mask, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        _, th1 = cv2.threshold(grad, 127, 255, cv2.THRESH_BINARY)

        a = []
        for row in range(distance - 50, distance + 50):
            if np.where(th1[row] == 255)[0].size == 0:
                continue
            else:
                if left_line:
                    a.append(np.where(th1[row] == 255)[0][-1])
                else:
                    a.append(np.where(th1[row] == 255)[0][0])

        if len(a) > 10:
            return np.median(a)
        else:
            return no_lane_value

    def get_x_center_for_color(self, mask, distance):
        """
        Returns the x-center of the detected color pixels at the given row distance.
        Returns None if no pixels are found.
        """
        row_start = max(0, distance - 50)
        row_end   = min(mask.shape[0], distance + 50)
        roi  = mask[row_start:row_end, :]
        cols = np.where(roi > 0)[1]
        if len(cols) > 0:
            return int(np.median(cols))
        return None

    def cbFindLane(self, image_msg):

        if self.counter <= 3:
            self.counter += 1
            return

        if self.is_running:
            return

        self.is_running = True
        self.counter = 0

        np_arr   = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Write frame to video if recording is enabled
        if self.video_record_enable:
            if self.video_writer is None:
                h, w = cv_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                os.makedirs(os.path.dirname(self.video_out_path), exist_ok=True)
                self.video_writer = cv2.VideoWriter(
                    self.video_out_path, fourcc, self.fps, (w, h)
                )
                if self.video_writer.isOpened():
                    print(f"[VIDEO] Recording: {w}x{h} @ {self.fps}fps -> {self.video_out_path}")
                else:
                    print(f"[VIDEO] Failed to open VideoWriter at: {self.video_out_path}")
                    self.video_writer = None
                    self.video_record_enable = False

            if self.video_writer is not None and self.video_writer.isOpened():
                self.video_writer.write(cv_image)

        img = self.crop_img(cv_image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- White mask: HSV thresholding (YOLO disabled) ---
        mask_white = cv2.inRange(
            hsv,
            (self.hue_white_l, self.saturation_white_l, self.lightness_white_l),
            (self.hue_white_h, self.saturation_white_h, self.lightness_white_h),
        )

        # --- Yellow mask: HSV thresholding ---
        mask_yellow = cv2.inRange(
            hsv,
            (self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l),
            (self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h),
        )

        # --- Red mask: two HSV ranges to cover the hue wrap-around ---
        mask_red1 = cv2.inRange(
            hsv,
            (self.hue_red_l, self.saturation_red_l, self.lightness_red_l),
            (self.hue_red_h, self.saturation_red_h, self.lightness_red_h),
        )
        mask_red2 = cv2.inRange(
            hsv,
            (170, self.saturation_red_l, self.lightness_red_l),
            (180, self.saturation_red_h, self.lightness_red_h),
        )
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Remove red pixels from lane masks so the red-line edge
        # cannot confuse the Sobel detector
        red_inverted = cv2.bitwise_not(mask_red)
        mask_white  = cv2.bitwise_and(mask_white,  red_inverted)
        mask_yellow = cv2.bitwise_and(mask_yellow, red_inverted)

        # Publish red pixel count — only bottom 10% of image (close range)
        bottom_start    = int(len(img) * 0.9)
        red_pixel_count = self.get_red_pixel_count(mask_red[bottom_start:])
        msg_red      = Float64()
        msg_red.data = float(red_pixel_count)
        self.pub_red_line.publish(msg_red)

        white_alternative  = int(len(img[0]) * 0.95)
        yellow_alternative = int(len(img[0]) * 0.05)

        center_white  = self.get_x_for_driving(mask_white,  int(len(img) * 0.75), white_alternative,  left_line=True)
        center_yellow = self.get_x_for_driving(mask_yellow, int(len(img) * 0.75), yellow_alternative, left_line=False)

        if center_white <= center_yellow:
            if center_white > int(len(img[0]) * 0.4):
                center_yellow = yellow_alternative
            else:
                center_white = white_alternative

        lane_center = (center_white + center_yellow) / 2

        msg_error      = Float64()
        msg_error.data = 1 - (lane_center / len(img) * 2)
        self.pub_lane.publish(msg_error)

        # Red line center position in lower row (for debug overlay)
        detection_row = int(len(img) * 0.95)
        center_red    = self.get_x_center_for_color(mask_red, detection_row)

        # Save state for debug / main-thread display
        self.img              = img
        self.lane_center      = lane_center
        self.white_alternative  = white_alternative
        self.yellow_alternative = yellow_alternative
        self.center_white     = center_white
        self.center_yellow    = center_yellow
        self.center_red       = center_red
        self.debug_img_white  = mask_white
        self.debug_img_yellow = mask_yellow
        self.debug_img_red    = mask_red

        # Build annotated display image
        image = cv2.circle(img, (int(lane_center), int(len(img) / 2)), 3, (255, 0, 0))
        image = cv2.line(image, (white_alternative,  0), (white_alternative,  self._crop_im_size), color=(255, 255, 255))
        image = cv2.line(image, (yellow_alternative, 0), (yellow_alternative, self._crop_im_size), color=(255, 255, 0))
        image = cv2.line(image, (0, int(len(img) * 0.75) + 100), (len(img[0]), int(len(img) * 0.75) + 100), color=(255, 255, 255))
        image = cv2.line(image, (0, int(len(img) * 0.75) - 100), (len(img[0]), int(len(img) * 0.75) - 100), color=(255, 255, 255))
        image = cv2.line(image, (int(len(img[0]) / 2), 0), (int(len(img[0]) / 2), len(image)), (0, 255, 0))
        image = cv2.circle(image, (int(center_white),  int(len(img) * 0.75)), 5, (255, 255, 255))
        image = cv2.circle(image, (int(center_yellow), int(len(img) * 0.75)), 5, (0, 255, 255))

        if center_red is not None:
            image = cv2.circle(image, (center_red, detection_row), 5, (0, 0, 255))

        if self.video_record_enable:
            cv2.putText(image, "REC",              (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(image, "R=record  Q=stop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self.display_image = image
        self.is_running    = False

    def fnShutDown(self):
        rospy.loginfo("Shutting down. Closing video file...")
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.release()
        cv2.destroyAllWindows()

    def run_debug(self):
        rospy.on_shutdown(self.fnShutDown)
        rate = rospy.Rate(10)
        cv2.namedWindow('lane detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('lane detection', cv2.WND_PROP_TOPMOST, 1)

        while not rospy.is_shutdown():

            if self.display_image is not None:
                cv2.imshow('lane detection', self.display_image)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('r') or key == ord('R'):
                if not self.video_record_enable:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.video_out_path  = f"/workspace/src/videos/lane_{timestamp}.mp4"
                    self.video_record_enable = True
                    print(f"[VIDEO] Recording STARTED: {self.video_out_path}")
            elif key == ord('q') or key == ord('Q'):
                if self.video_record_enable:
                    self.video_record_enable = False
                    if self.video_writer is not None and self.video_writer.isOpened():
                        self.video_writer.release()
                        self.video_writer = None
                    print(f"[VIDEO] Recording STOPPED")

            # Publish debug topics (only when subscribers are connected)
            if hasattr(self, 'img'):

                if self.pub_debug_lane.get_num_connections() > 0:
                    debug_img = self.img.copy()
                    debug_img = cv2.circle(debug_img, (int(self.lane_center), int(len(debug_img) / 2)), 3, (255, 0, 0))
                    debug_img = cv2.line(debug_img, (self.white_alternative,  0), (self.white_alternative,  1000), color=(255, 255, 255))
                    debug_img = cv2.line(debug_img, (self.yellow_alternative, 0), (self.yellow_alternative, 1000), color=(255, 255, 0))
                    debug_img = cv2.line(debug_img, (0, int(len(debug_img) * 0.75) + 100), (len(debug_img[0]), int(len(debug_img) * 0.75) + 100), color=(255, 255, 255))
                    debug_img = cv2.line(debug_img, (0, int(len(debug_img) * 0.75) - 100), (len(debug_img[0]), int(len(debug_img) * 0.75) - 100), color=(255, 255, 255))
                    debug_img = cv2.line(debug_img, (int(len(debug_img[0]) / 2), 0), (int(len(debug_img[0]) / 2), len(debug_img)), (0, 255, 0))
                    debug_img = cv2.circle(debug_img, (int(self.center_white),  int(len(debug_img) * 0.75)), 5, (255, 255, 255))
                    debug_img = cv2.circle(debug_img, (int(self.center_yellow), int(len(debug_img) * 0.75)), 5, (0, 255, 255))

                    if self.center_red is not None:
                        detection_row = int(len(debug_img) * 0.75)
                        debug_img = cv2.circle(debug_img, (self.center_red, detection_row), 5, (0, 0, 255))

                    if self.video_record_enable:
                        cv2.putText(debug_img, "REC",              (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(debug_img, "R=record  Q=stop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    debug_msg = CompressedImage()
                    debug_msg.header.stamp = rospy.Time.now()
                    debug_msg.format = "jpeg"
                    debug_msg.data   = np.array(cv2.imencode('.jpg', debug_img)[1]).tobytes()
                    self.pub_debug_lane.publish(debug_msg)

                if self.pub_debug_white.get_num_connections() > 0:
                    debug_msg = CompressedImage()
                    debug_msg.header.stamp = rospy.Time.now()
                    debug_msg.format = "jpeg"
                    debug_msg.data   = np.array(cv2.imencode('.jpg', self.debug_img_white)[1]).tobytes()
                    self.pub_debug_white.publish(debug_msg)

                if self.pub_debug_yellow.get_num_connections() > 0:
                    debug_msg = CompressedImage()
                    debug_msg.header.stamp = rospy.Time.now()
                    debug_msg.format = "jpeg"
                    debug_msg.data   = np.array(cv2.imencode('.jpg', self.debug_img_yellow)[1]).tobytes()
                    self.pub_debug_yellow.publish(debug_msg)

                if self.pub_debug_red.get_num_connections() > 0:
                    debug_msg = CompressedImage()
                    debug_msg.header.stamp = rospy.Time.now()
                    debug_msg.format = "jpeg"
                    debug_msg.data   = np.array(cv2.imencode('.jpg', self.debug_img_red)[1]).tobytes()
                    self.pub_debug_red.publish(debug_msg)

            rate.sleep()


if __name__ == '__main__':
    node = DetectLaneNode('detect_lane_node')
    node.run_debug()
    rospy.spin()