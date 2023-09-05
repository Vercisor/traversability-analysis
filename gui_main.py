import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QWidget, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint

from robomaster_controller import RobomasterController
from keras.models import load_model
import queue
from robomaster import robot
from multiprocessing import Queue, Process
import skimage.transform

import numpy as np

class CameraThread(QThread):
    def __init__(self, camera, label):
        QThread.__init__(self)
        self.camera = camera
        self.label = label

    def run(self):
        while True:
            ret, frame = self.camera.read()

            if frame is None or frame.size == 0:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (448, 448))

            height, width, _ = frame.shape
            bytes_per_line = 3 * width
            image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            self.label.setPixmap(QPixmap.fromImage(image))

class InferenceProcess(Process):
    def __init__(self, model, vision_queue, inference_queue, mask_queue):
        Process.__init__(self)
        self.model = model
        self.vision_queue = vision_queue
        self.inference_queue = inference_queue
        self.mask_queue = mask_queue

    def run(self):
        while True:
            try:
                img = self.vision_queue.get(block=False)
                img = cv2.resize(img, (224, 224)) / 255
                pred = self.model.predict(img.reshape(1, 224, 224, 3))[0]
                self.inference_queue.put(pred, block=False)
                self.mask_queue.put(pred, block=False)
            except queue.Empty:
                pass
            except queue.Full:
                pass

class MaskThread(QThread):
    update_mask_warp_signal = pyqtSignal(QPixmap)
    def __init__(self, inference_queue, command_queue, label):
        QThread.__init__(self)
        self.queue: Queue = inference_queue
        self.command_queue: Queue = command_queue
        self.label = label

    def warp_image(self, img):
        A = np.array([62, 31])
        B = np.array([62, -31])
        C = np.array([124, -31])
        D = np.array([124, 31])

        tf_if = skimage.transform.estimate_transform("projective",
                                                    src= np.float32([[58, 188], [180, 188], [149, 148], [87, 148]]),
                                                    dst= np.vstack((A,B,C,D)))
        tf_fo = skimage.transform.estimate_transform("projective",
                                                    src=np.float32([[0, 100], [0, -100], [200, -100], [200, 100]]),
                                                    dst=np.float32([[0, 1000], [1000, 1000], [1000, 0], [0, 0]]),
                                                    )
        tf = (tf_if + tf_fo)

        tf_im = skimage.transform.warp(image=img,
                                inverse_map=tf.inverse,
                                output_shape=(1000, 1000))
        
        tf_im = np.array(tf_im)

        tf_im = (tf_im * 255).astype(np.uint8)

        #warped_image = cv2.cvtColor(tf_im, cv2.COLOR_BGR2RGB)

        return tf_im
    
    def draw_trapezoid(self, img, trapezoid, color):
        pts = np.array(trapezoid, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color)

    def draw_rectangles(self, img, color=(255, 255, 255)):
        self.draw_trapezoid(img, [(33, 223),(87, 148),(149, 148),(205, 223)], color)
        self.draw_trapezoid(img, [(0, 223),(33, 223),(87, 148),(0, 148)], color)
        self.draw_trapezoid(img, [(205,223),(149, 148),(223, 148),(223, 223)], color)

    def draw_direction(self, direction, img):
        if direction == "front":
            self.draw_trapezoid(img, [(33, 223),(87, 148),(149, 148),(205, 223)], (0, 255, 0))
        elif direction == "left":
            self.draw_trapezoid(img, [(0, 223),(33, 223),(87, 148),(0, 148)], (0, 255, 0))
        elif direction == "right":
            self.draw_trapezoid(img, [(205,223),(149, 148),(223, 148),(223, 223)], (0, 255, 0))
        else:
            pass

    def run(self):
        while True:
            try:
                mask = self.queue.get(block=False)
            except:
                continue

            if mask is None:
                continue

            # Rescale pixel values to range 0-255 and convert to 3-channel color
            mask = cv2.cvtColor((mask * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

            warped_mask = self.warp_image(mask)
            warped_mask = cv2.resize(warped_mask, (300, 300))

            self.draw_rectangles(mask)

            try:
                direction, counts = self.command_queue.get(block=False)
                self.draw_direction(direction, mask)
            except:
                pass

            mask = cv2.resize(mask, (300, 300))

            height, width, _ = mask.shape
            bytes_per_line = 3 * width  # Three channels in color image
            image = QImage(mask.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)  # Color format

            warped_msk = QImage(warped_mask.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            warped_pixmap = QPixmap.fromImage(warped_msk)

            self.label.setPixmap(QPixmap.fromImage(image))
            self.update_mask_warp_signal.emit(warped_pixmap)

class StreamThread(QThread):
    update_text_labels_signal = pyqtSignal(str, str, str)
    update_image_warp_signal = pyqtSignal(QPixmap)
    def __init__(self, camera, vision_queue, command_queue, label):
        QThread.__init__(self)
        self.camera = camera
        self.command_queue: Queue = command_queue
        self.vision_queue: Queue = vision_queue
        self.label = label
    
    def overlay_image(self, main_image, overlay_image, position=(0, 0)):
        x, y = position

        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)
        # Extract the BGR and alpha channels from the overlay image
        overlay_BGR = overlay_image[..., :3]
        alpha = overlay_image[..., 3] / 255.0

        # Resize the BGR and alpha channels to match the desired size of the overlay
        h, w = 12, 36
        overlay_BGR = cv2.resize(overlay_BGR, (w, h))
        alpha = cv2.resize(alpha, (w, h))
        alpha_3channel = cv2.merge((alpha, alpha, alpha))  # Convert alpha to 3-channel

        # Define the region of interest (ROI) in the main image
        roi = main_image[y:y+h, x:x+w]

        # Check if the ROI and overlay image dimensions match
        if roi.shape[:2] == overlay_BGR.shape[:2]:
            # Blend the overlay image with the ROI using the alpha channel
            blended_roi = (1.0 - alpha_3channel) * roi + alpha_3channel * overlay_BGR
            main_image[y:y+h, x:x+w] = blended_roi.astype(np.uint8)

        return main_image

    def warp_image(self, img):
        A = np.array([62, 31])
        B = np.array([62, -31])
        C = np.array([124, -31])
        D = np.array([124, 31])

        tf_if = skimage.transform.estimate_transform("projective",
                                                    src= np.float32([[58, 188], [180, 188], [149, 148], [87, 148]]),
                                                    dst= np.vstack((A,B,C,D)))
        tf_fo = skimage.transform.estimate_transform("projective",
                                                    src=np.float32([[0, 100], [0, -100], [200, -100], [200, 100]]),
                                                    dst=np.float32([[0, 1000], [1000, 1000], [1000, 0], [0, 0]]),
                                                    )
        tf = (tf_if + tf_fo)

        tf_im = skimage.transform.warp(image=img,
                                inverse_map=tf.inverse,
                                output_shape=(1000, 1000))
        
        tf_im = np.array(tf_im)

        tf_im = (tf_im * 255).astype(np.uint8)

        #warped_image = cv2.cvtColor(tf_im, cv2.COLOR_BGR2RGB)

        return tf_im
    
    def draw_trapezoid(self, img, trapezoid, color):
        pts = np.array(trapezoid, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color)

    def draw_rectangles(self, img, color=(255, 255, 255)):
        self.draw_trapezoid(img, [(33, 223),(87, 148),(149, 148),(205, 223)], color)
        self.draw_trapezoid(img, [(0, 223),(33, 223),(87, 148),(0, 148)], color)
        self.draw_trapezoid(img, [(205,223),(149, 148),(223, 148),(223, 223)], color)

    def draw_direction(self, direction, img):
        if direction == "front":
            self.draw_trapezoid(img, [(33, 223),(87, 148),(149, 148),(205, 223)], (0, 255, 0))
        elif direction == "left":
            self.draw_trapezoid(img, [(0, 223),(33, 223),(87, 148),(0, 148)], (0, 255, 0))
        elif direction == "right":
            self.draw_trapezoid(img, [(205,223),(149, 148),(223, 148),(223, 223)], (0, 255, 0))
        else:
            pass

    def run(self):
        self.camera.start_video_stream(display=False)
        counts = {"front": 0.0, "left": 0.0, "right": 0.0}
        while True:
            frame = self.camera.read_cv2_image(strategy="newest")

            try:
                self.vision_queue.put(frame)
            except:
                pass

            if frame is None or frame.size == 0:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            self.draw_rectangles(frame)
            

            try:
                direction, counts = self.command_queue.get(block=False)
                self.draw_direction(direction, frame)
                self.update_text_labels_signal.emit(str(np.round(counts["left"], 2)), 
                                                    str(np.round(counts["front"], 2)), 
                                                    str(np.round(counts["right"], 2)))
            except:
                pass
            warped_image = self.warp_image(frame)
            
            warped_image = cv2.resize(warped_image, (300, 300))
            
            # Read and resize the overlay image
            overlay_img = cv2.imread('new_gui/img/Robomaster_camera.png', cv2.IMREAD_UNCHANGED)
            position = (132, 288)

            overlaid_image = self.overlay_image(main_image=warped_image, overlay_image=overlay_img, position=position)

            frame = cv2.resize(frame, (300, 300))
            height, width, _ = frame.shape
            bytes_per_line = 3 * width
            img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(img)

            warped_img = QImage(overlaid_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            warped_pixmap = QPixmap.fromImage(warped_img)

            self.label.setPixmap(pixmap)
            self.update_image_warp_signal.emit(warped_pixmap)


class MainWindow(QMainWindow):
    def __init__(self, camera_indices=[0, 1, 2], *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.model = load_model('models/unet_aug_fine_tuned_uni.hdf5')
        self.inference_queue = Queue(maxsize=1)
        self.vision_queue = Queue(maxsize=1)
        self.mask_queue = Queue(maxsize=1)
        self.command_queue_img = Queue(maxsize=1)
        self.command_queue_mask = Queue(maxsize=1)
        self.inference_process = InferenceProcess(self.model,
                                                  self.vision_queue,
                                                  self.inference_queue, 
                                                  self.mask_queue)

        self.robot = robot.Robot()
        self.robot.initialize(conn_type="ap")
        self.camera = self.robot.camera
        self.robot.set_robot_mode(robot.CHASSIS_LEAD)
        self.controller = RobomasterController(1, self.robot, self.mask_queue, self.command_queue_img, self.command_queue_mask)
        self.layout = QHBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.webcam = cv2.VideoCapture(2)
        self.labels = [QLabel() for _ in camera_indices]
        self.text_labels = [QLabel() for _ in range(3)]  # create text labels
        for text_label in self.text_labels:
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for i, label in enumerate(self.labels):
            if i == 1:  # for the second image
                vbox = QVBoxLayout()
                vbox.addWidget(label)
                # add image that will be updated here
                self.image_warp = QLabel()
                self.image_warp.setAlignment(Qt.AlignmentFlag.AlignCenter)
                hbox = QHBoxLayout()
                vbox.addLayout(hbox)
                for text_label in self.text_labels:  # add three text labels
                    hbox.addWidget(text_label)
                vbox.addWidget(self.image_warp)
                self.layout.addLayout(vbox)
            elif i == 2:
                vbox = QVBoxLayout()
                vbox.addWidget(label)
                self.mask_warp = QLabel()
                self.mask_warp.setAlignment(Qt.AlignmentFlag.AlignCenter)
                vbox.addWidget(self.mask_warp)
                self.layout.addLayout(vbox)
            else:
                self.layout.addWidget(label)

        self.webcam = CameraThread(self.webcam, self.labels[0])
        self.image = StreamThread(self.camera, self.vision_queue, self.command_queue_img, self.labels[1])
        self.image.update_text_labels_signal.connect(self.update_text_labels)  # connect the signal to a slot

        self.image.update_image_warp_signal.connect(self.image_warp.setPixmap)  # connect the signal to a slot

        self.inferred_mask = MaskThread(self.inference_queue, self.command_queue_mask, self.labels[2])

        self.inferred_mask.update_mask_warp_signal.connect(self.mask_warp.setPixmap) 

        self.threads = [self.webcam, self.image, self.inferred_mask]

        self.inference_process.start()

        for thread in self.threads:
            thread.start()
        
        self.controller.start()
    
    def update_text_labels(self, text1, text2, text3):
        self.text_labels[0].setText(text1)
        self.text_labels[1].setText(text2)
        self.text_labels[2].setText(text3)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()