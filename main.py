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
        self.detected_cones = DetectedCones(turtle)
        self.new_detected_cones = None
        self.angle = None
        self.distance = None
        self.counter = 1
        self.center = None
        self.angle_to_turn = None

    def run_state(self):
        self.current_state()

    def idle(self):
        print("IDLE")
        self.current_state = self.detect_cones
        self.turtle.cmd_velocity(linear=0.0, angular=0)

    def look_around1(self):
        self.new_detected_cones = None
        print(self.turtle.get_odometry()[2])
        if self.turtle.get_odometry()[2] > -np.pi / 2:
            self.turtle.cmd_velocity(linear=0, angular=-0.8)
        else:
            time.sleep(0.4)
            self.detect_cones()
            time.sleep(0.2)
            self.current_state = self.look_around2

    def look_around2(self):
        self.new_detected_cones = None

        if self.turtle.get_odometry()[2] < -np.pi / 2 + (np.pi / 9) * self.counter:
            self.turtle.cmd_velocity(linear=0, angular=0.65)
        else:
            self.turtle.cmd_velocity(linear=0, angular=0)
            time.sleep(0.2)
            self.detect_cones()
            if self.counter > 8:
                self.current_state = self.estimate_cones_position
            else:
                self.counter += 1
                self.current_state = self.look_around2

    def detect_cones(self):
        point_cloud = self.turtle.get_point_cloud()
        rgb_image = self.turtle.get_rgb_image()
        image_copy = rgb_image.copy()
        self.new_detected_cones = DetectedCones(self.turtle)  # -> detectedCones.red, green, blue
        self.new_detected_cones.detect_cones(rgb_image, point_cloud)
        self.new_detected_cones.draw_cones(
            image_copy)  # -> az na konec, prekresli puvodni obrazek mohlo by se s nim pak hure pracovat
        cv2.imshow("RGB", image_copy)
        self.merge_new_cones()
        # self.current_state = self.estimate_cones_position

    def merge_new_cones(self):
        print(self.new_detected_cones.all)
        for new_cone in self.new_detected_cones.all:
            if new_cone not in self.detected_cones.all:
                self.detected_cones.add_cone(new_cone)

    def estimate_cones_position(self):
        pair = self.detected_cones.get_closest_pair()
        if pair is not None:
            print()
            first = (pair[0].distance * np.sin(pair[0].odo - pair[0].angle),
                     pair[0].distance * np.cos(pair[0].odo - pair[0].angle))
            second = (pair[1].distance * np.sin(pair[1].odo - pair[1].angle),
                     pair[1].distance * np.cos(pair[1].odo - pair[1].angle))
            print("first ",first," second ",second)


            self.center = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
            hypotenuse = ((second[1] - first[1]) / 2)**2 + ((first[0] - second[0]) / 2)**2
            scale = 0.5/hypotenuse
            approach_point_deviation = [(second[1] - first[1]) / (2*scale),(first[0] - second[0]) / (2*scale)]
            goal1 = (self.center[0] + approach_point_deviation[0], self.center[1] + approach_point_deviation[1])
            goal2 = (self.center[0] - approach_point_deviation[0], self.center[1] - approach_point_deviation[1])
            print("goal1", goal1, "goal2", goal2)
            dist1 = np.sqrt(goal1[0] ** 2 + goal1[1] ** 2)
            dist2 = np.sqrt(goal2[0] ** 2 + goal2[1] ** 2)
            if dist1 < dist2:
                self.angle = np.arcsin(goal1[0] / dist1)
                self.distance = dist1
            else:
                self.angle = -np.arcsin(goal2[0] / dist2)
                self.angle = -self.angle
                self.distance = dist2
            
            print(self.angle, dist1, dist2)
            self.distance -= 0.1
            #elf.distance -= self.distance * 0.06
            self.current_state = self.turn_to_middle

    def turn_to_middle(self):
        print(self.turtle.get_odometry()[2])
        if self.turtle.get_odometry()[2] < -0.05:
            self.turtle.cmd_velocity(linear=0, angular=0.4)
        elif self.turtle.get_odometry()[2]> 0.05:
            self.turtle.cmd_velocity(linear=0, angular=-0.4)
        else:
            time.sleep(0.7)
            self.current_state = self.turn_turtle_to_angle

    def turn_turtle_to_angle(self):
        if self.turtle.get_odometry()[2] < self.angle - 0.05:
            self.turtle.cmd_velocity(linear=0, angular=0.4)
        elif self.turtle.get_odometry()[2] > self.angle + 0.05:
            self.turtle.cmd_velocity(linear=0, angular=-0.4)
        else:
            self.turtle.reset_odometry()
            self.current_state = self.drive_turtle_to_position

    def drive_turtle_to_position(self):
        odom = self.turtle.get_odometry()
        if np.sqrt(odom[0] ** 2 + odom[1] ** 2) < self.distance:
            self.turtle.cmd_velocity(linear=0.25, angular=0)
        else:
            self.current_state = self.calc_turn_to_goal

    def calc_turn_to_goal(self):
        self.distance += 0.1
        center_dist = self.center[0]**2 + self.center[1]**2
        parameter1 = self.distance**2 + (0.25*np.sqrt(3))**2 - center_dist
        parameter2 = 2*self.distance*(0.25*np.sqrt(3))
        self.angle_to_turn =np.pi- np.arccos(parameter1/parameter2)
        if self.angle>0:
            self.angle_to_turn = -self.angle_to_turn
        self.current_state = self.turn_to_goal

    def turn_to_goal(self):
        if self.turtle.get_odometry()[2] < self.angle_to_turn - 0.05:
            self.turtle.cmd_velocity(linear=0, angular=0.3)
        elif self.turtle.get_odometry()[2] > self.angle_to_turn + 0.05:
            self.turtle.cmd_velocity(linear=0, angular=-0.3)
        else:
            self.turtle.reset_odometry()
            self.current_state = self.drive_through

    def drive_through(self):
        odom = self.turtle.get_odometry()
        if np.sqrt(odom[0] ** 2 + odom[1] ** 2) <  1:
            self.turtle.cmd_velocity(linear=1, angular=0)
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
    turtle.reset_odometry()
    plt.ion()

    while not turtle.is_shutting_down():
        state_machine.run_state()
        if state_machine.detected_cones is not None:
            for cone in state_machine.detected_cones.all:
                if cone.color is Color.RED:
                    plt.scatter(cone.odo-cone.angle, cone.distance, s=40, color='red')
                if cone.color is Color.GREEN:
                    plt.scatter(cone.odo-cone.angle, cone.distance, s=40, color='green')
                if cone.color is Color.BLUE:
                    plt.scatter(cone.odo-cone.angle, cone.distance, s=40, color='blue')
        plt.pause(0.001)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
