from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate
from robolab_turtlebot import detector, get_time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

from cone_detections import DetectedCones
from cone import Cone, Color

x_range = (-0.3, 0.3)
z_range = (0.3, 3.0)

WINDOW_D = 'DEPTH'  # depth
WINDOW = 'RGB'

stop = False
t = 0
fun_step = 0


class StateMachine:
    def __init__(self, turtle):
        self.current_state = self.look_around1
        self.turtle = turtle
        self.detected_cones = None
        self.angle = None
        self.distance = None
        self.look_around_angle = 90

    def run_state(self):
        self.current_state()
        self.detect_cones()

    def idle(self):
        print("IDLE")
        self.turtle.cmd_velocity(linear=0.0, angular=0)

    def look_around1(self):
        print(self.turtle.get_odometry()[2])
        if self.turtle.get_odometry()[2] > -np.pi/2:
            self.turtle.cmd_velocity(linear=0, angular=-0.3)
        else:
            self.current_state = self.look_around2


    def look_around2(self):
        if self.turtle.get_odometry()[2] < np.pi / 2:
            self.turtle.cmd_velocity(linear=0, angular=0.3)
        else:
            self.current_state = self.idle

    def detect_cones(self):
        point_cloud = self.turtle.get_point_cloud()
        rgb_image = self.turtle.get_rgb_image()
        image_copy = rgb_image.copy()
        self.detected_cones = DetectedCones()  # -> detectedCones.red, green, blue
        self.detected_cones.detect_cones(rgb_image, point_cloud)
        self.detected_cones.draw_cones(
            image_copy)  # -> az na konec, prekresli puvodni obrazek mohlo by se s nim pak hure pracovat
        cv2.imshow("RGB", image_copy)
        #self.current_state = self.estimate_cones_position

    def estimate_cones_position(self):
        pair = self.detected_cones.get_closest_pair()
        if len(pair) > 1:
            center = ((pair[0].x + pair[1].x) / 2, (pair[0].y + pair[1].y) / 2)
            goal1 = (center[0] + (pair[1].y - pair[0].y) / 2, center[1] + (pair[0].x - pair[1].x) / 2)
            goal2 = (center[0] - (pair[1].y - pair[0].y) / 2, center[1] - (pair[0].x - pair[1].x) / 2)
            print("goal1", goal1, "goal2", goal2)
            dist1 = np.sqrt(goal1[0] ** 2 + goal1[1] ** 2)
            dist2 = np.sqrt(goal2[0] ** 2 + goal2[1] ** 2)
            if dist1 < dist2:
                self.angle = np.pi / 2 - np.arcsin(goal1[0] / dist1)
                distance = dist1
            else:
                self.angle = np.arcsin(goal2[0] / dist2) - np.pi / 2
                distance = dist2
            print(self.angle, dist1, dist2)
            self.turtle.reset_odometry()
            self.distance -= distance * 0.05
            self.current_state = self.turn_turtle_to_angle

    def turn_turtle_to_angle(self):
        if self.turtle.get_odometry()[2] < self.angle - 0.05:
            self.turtle.cmd_velocity(linear=0, angular=0.3)
        elif self.turtle.get_odometry()[2] > self.angle + 0.05:
            self.turtle.cmd_velocity(linear=0, angular=-0.3)
        else:
            self.turtle.reset_odometry()
            self.current_state = self.drive_turtle_to_position

    def drive_turtle_to_position(self):
        odom = self.turtle.get_odometry()
        if np.sqrt(odom[0] ** 2 + odom[1] ** 2) < self.distance:
            self.turtle.cmd_velocity(linear=0.3, angular=0)
        else:
            self.current_state = self.idle


def fun(turtle):
    # global fun_step
    # fun_step += 1
    # fun_step %= 7
    # turtle.play_sound(fun_step)
    global t
    print(turtle.get_odometry())
    if abs(turtle.get_odometry()[2]) < np.pi:
        turtle.cmd_velocity(linear=0, angular=1)
    else:
        turtle.cmd_velocity(linear=0, angular=0)


# stop robot
def bumper_cb(msg):
    global stop
    global t
    t = get_time()

    stop = True
    print('Bumper was activated, new state is STOP')


def main():
    global stop
    turtle = Turtlebot(pc=True, rgb=True, depth=True)
    state = 0
    cv2.namedWindow(WINDOW)  # display rgb image
    turtle.register_bumper_event_cb(bumper_cb)
    angle = 0
    distance = 0
    state_machine = StateMachine(turtle)
    plt.ion()

    while not turtle.is_shutting_down():
        state_machine.run_state()
        for cone in state_machine.detected_cones.all:
            if cone.color is Color.RED:
                plt.scatter(cone.x, cone.y, s=10,color='red')
            if cone.color is Color.GREEN:
                plt.scatter(cone.x, cone.y, s=10,color='green')
            if cone.color is Color.BLUE:
                plt.scatter(cone.x, cone.y, s=10,color='blue')
        plt.show()
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
