#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        # Setup camera
        self._cam_id = 0
        self._cam_fps = 60
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        # Setup SIFT
        self.sift = cv2.SIFT_create()
        self.index_params = dict(algorithm=0, trees=5)
        self.search_params = dict()
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.kp_image = None
        self.desc_image = None

    def sift_image(self):
        grayframe = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)
        matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
        # Find the Homography of the train image
        try:
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            # Perform a perspective transform on the homography
            h, w = self.img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            # Print the homography to the image
            homography = cv2.polylines(self.frame, [np.int32(dst)], True, (255, 0, 0), 3)
            return homography
        except:
            return self.frame

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
        pixmap = QtGui.QPixmap(self.template_path)
        # Store the image as a cv2 image + detect the SIFT features
        self.img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)
    
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                     bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, self.frame = self.video_reader.read()
        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        if ret:
            # Perform SIFT analysis only if there is a template image loaded
            if self.desc_image is not None:
                edited_frame = self.sift_image()
                pixmap = self.convert_cv_to_pixmap(cv2.resize(edited_frame, (240, 320)))
            else:
                pixmap = self.convert_cv_to_pixmap(cv2.resize(self.frame, (240, 320)))
            self.live_image_label.setPixmap(pixmap)
    
    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            dlg = QtWidgets.QFileDialog()
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if dlg.exec_():
                self.video_path = dlg.selectedFiles()[0]
            self.video_reader = cv2.VideoCapture(self.video_path)
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())