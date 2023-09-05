import queue
from threading import Thread
import cv2
import numpy as np
from keras.models import load_model
import multiprocessing
from robomaster import robot, led

class VisionController(Thread):
    def __init__(self, camera, vision_queue, command_queue, master_queue, model):
        Thread.__init__(self)
        self.camera = camera
        self.vision_queue: multiprocessing.Queue = vision_queue
        self.command_queue: queue.Queue = command_queue
        self.master_queue: queue.Queue = master_queue
        self.model = model

    def draw_rectangle(self, img, rectangle, color):
        (x1, y1), (x2, y2) = rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color)

    def draw_rectangles(self, img, color=(255, 255, 255)):
        self.draw_rectangle(img, [(160,200),(64, 100)], color)
        self.draw_rectangle(img, [(64,200),(0, 100)], color)
        self.draw_rectangle(img, [(160,200),(223, 100)], color)

    def draw_direction(self, direction, img):
        if direction == "front":
            self.draw_rectangle(img, [(160,200),(64, 100)], (0, 255, 0))
        elif direction == "left":
            self.draw_rectangle(img, [(64,200),(0, 100)], (0, 255, 0))
        elif direction == "right":
            self.draw_rectangle(img, [(160,200),(223, 100)], (0, 255, 0))
        else:
            pass

    def run(self):
        self.camera.start_video_stream(display=False)
        counts = {"front": 0.0, "left": 0.0, "right": 0.0}
        while True:
            img = self.camera.read_cv2_image(strategy="newest")
            try:
                self.vision_queue.put(img, block=False)
            except queue.Full:
                pass
            try: 
                command_dict = self.command_queue.get(block=False)
                direction = command_dict["direction"]
                counts = command_dict["counts"]
                pred = command_dict["pred"]
                img = command_dict["img"]
                self.draw_rectangles(img)
                self.draw_direction(direction, img)
                self.master_queue.put({"image": cv2.resize(img, (448, 448)), 
                                    "mask": pred, 
                                    "counts": counts}, block=False)
            except queue.Empty:
                pass
            except queue.Full:
                pass
        


class RobomasterController(Thread):
    
    def __init__(self, id, robot, mask_queue, command_queue_img, command_queue_mask):
        Thread.__init__(self)
        self.id = id
        self.ep_robot = robot
        self.command_queue_img: multiprocessing.Queue = command_queue_img
        self.command_queue_mask: multiprocessing.Queue = command_queue_mask
        self.connect()
        self.mask_queue: multiprocessing.Queue = mask_queue

    def choose_direction(self, pred, moves):
        speed = 0
        trapezoid_front = [(33, 223),(87, 148),(149, 148),(205, 223)]
        trapezoid_left = [(0, 223),(33, 223),(87, 148),(0, 148)]
        trapezoid_right = [(205,223),(149, 148),(223, 148),(223, 223)]
        trapezoids = [trapezoid_front, trapezoid_left, trapezoid_right]

        directions = ["front", "left", "right"]
        counts = {}

        for i, trapezoid in enumerate(trapezoids):
            mask = np.zeros_like(pred)
            cv2.fillPoly(mask, [np.array(trapezoid)], 1)
            roi = pred * mask

            length = np.sum(mask)
            count = np.sum(roi > 0.5) / length
            counts[directions[i]] = count

        best_direction = min(counts, key=counts.get)

        # if the count of the best direction is more than 0.4

        if counts[best_direction] > 0.35:
            speed = -speed
            best_direction = "error"

        return best_direction, counts, speed

    def check_for_repetitions(self, moves):
        last_moves = moves[-4:]
        return last_moves[0] == last_moves[2] and last_moves[1] == last_moves[3]
    
    def set_led(self, direction, effect):
        if direction == "front":
            self.led.set_led(led.COMP_BOTTOM_FRONT, r=0, g=255, b=0, effect=effect)
            self.led.set_led(led.COMP_BOTTOM_BACK, r=0, g=255, b=0, effect=effect)
        elif direction == "left":
            self.led.set_led(led.COMP_BOTTOM_LEFT, r=0, g=255, b=0, effect=effect)
            self.led.set_led(led.COMP_TOP_LEFT, r=0, g=255, b=0, effect=effect)
        elif direction == "right":
            self.led.set_led(led.COMP_BOTTOM_RIGHT, r=0, g=255, b=0, effect=effect)
            self.led.set_led(led.COMP_TOP_RIGHT, r=0, g=255, b=0, effect=effect)
        else:
            self.led.set_led(led.COMP_BOTTOM_FRONT, r=255, g=0, b=0, effect=effect)
            self.led.set_led(led.COMP_BOTTOM_BACK, r=255, g=0, b=0, effect=effect)
            self.led.set_led(led.COMP_BOTTOM_RIGHT, r=255, g=0, b=0, effect=effect)
            self.led.set_led(led.COMP_TOP_RIGHT, r=255, g=0, b=0, effect=effect)
            self.led.set_led(led.COMP_BOTTOM_LEFT, r=255, g=0, b=0, effect=effect)
            self.led.set_led(led.COMP_TOP_LEFT, r=255, g=0, b=0, effect=effect)

    def move(self, direction, speed):
        
        if direction == "front":
            self.ep_chassis.drive_wheels(w1=speed, w2=speed, w3=speed, w4=speed)
        elif direction == "left":
            self.ep_chassis.drive_wheels(w1=speed, w2=0, w3=0, w4=0)
        elif direction == "right":
            self.ep_chassis.drive_wheels(w1=0, w2=speed, w3=0, w4=0)
        else:
            self.ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        
        """
        else:
            self.ep_robot.set_robot_mode(robot.GIMBAL_LEAD)
            # rotate the gimbal 90 degrees
            self.gimbal.move(yaw=180, pitch=0, yaw_speed=60, pitch_speed=60).wait_for_completed()
            self.ep_robot.set_robot_mode(robot.CHASSIS_LEAD)
        """
            

    def run(self):
        while True:
            try:
                pred = self.mask_queue.get(block=False)
                direction, counts, speed = self.choose_direction(pred, self.moves)
                try:
                    self.command_queue_img.put((direction, counts), block=False)
                    self.command_queue_mask.put((direction, counts), block=False)
                except:
                    pass
                self.set_led(direction, led.EFFECT_ON)
                if self.moves and direction != self.moves[-1]:
                    self.set_led(self.moves[-1], led.EFFECT_OFF)
                self.move(direction, speed)

                self.moves.append(direction)
            except queue.Empty:
                pass

    def connect(self):
        self.gimbal = self.ep_robot.gimbal
        self.ep_camera = self.ep_robot.camera
        self.ep_chassis = self.ep_robot.chassis
        self.gimbal.recenter().wait_for_completed()
        self.led = self.ep_robot.led
        self.led.set_led(led.COMP_ALL, r=0, g=255, b=0, effect=led.EFFECT_OFF)
        self.moves = []
        self.is_connected = True