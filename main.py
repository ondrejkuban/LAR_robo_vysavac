from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate
from robolab_turtlebot import detector, get_time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

from cone_detections import DetectedCones
from cone import Cone, Color

WINDOW = 'RGB'
MIDDLE_DIST_PRESET = 0.5


class StateMachine:
    def __init__(self, turtle):
        self.current_state = self.turn_to_initial_angle
        self.turtle = turtle
        self.detected_cones = DetectedCones(turtle)
        self.new_detected_cones = None
        self.angle = None
        self.distance = None
        self.counter = 1
        self.center = None
        self.angle_to_turn = None
        self.ready_to_drive_through = False
        self.direction = 1
        self.alpha = None
        self.look_around_step = np.pi/9
        self.angle_before_turn = None
        self.fun_step = 0
        self.last_cone_color = Color.INVALID
        self.actual_cone_color = Color.INVALID
        self.bumper_error = False
        self.finish = False

    def run_state(self):
        self.current_state()

    def turn_to_desired_angle(self,angle, side, speed):
        # side    -,right   +,left
        if self.angle_before_turn is None:
            self.angle_before_turn = self.turtle.get_odometry()[2]
        if side == "right":
            if self.turtle.get_odometry()[2]- self.angle_before_turn > -angle:
                self.turtle.cmd_velocity(linear=0, angular=-speed)
                return False
        elif side == "left":
            if self.turtle.get_odometry()[2]- self.angle_before_turn < angle:
                self.turtle.cmd_velocity(linear=0, angular=speed)
                return False
        self.turtle.cmd_velocity(linear=0, angular=0)
        self.angle_before_turn = None
        return True

    def turn_to_initial_angle(self):
        self.new_detected_cones = None
        if self.ready_to_drive_through: #dosahl jsem bodu pul metri pred stredem
            if self.direction == 1:
                if self.turn_to_desired_angle(np.pi/4,"right",0.6):
                    self.parse_camera_data()
                    self.current_state = self.close_look_around
                    return
            elif self.direction == -1 or self.direction == 2:
                if self.turn_to_desired_angle(np.pi/4,"left",0.6):
                    self.parse_camera_data()
                    self.current_state = self.close_look_around
                    return
        elif self.turn_to_desired_angle(np.pi/2,"right",0.6):
            self.parse_camera_data()
            self.current_state = self.close_look_around

    def close_look_around(self):
        self.new_detected_cones = None
        turn_number = 6
        if self.direction == 2:
            turn_number = 5
        turn_side = "right"
        if self.direction == 1:
            turn_side = "left"
        if not self.ready_to_drive_through:
            turn_side = "left"
            turn_number = 7
        if self.turn_to_desired_angle(self.look_around_step,turn_side,0.6):
            self.parse_camera_data()
            if self.counter > turn_number:
                self.counter = 1
                self.current_state = self.calculate_move
            else:
                self.counter += 1
                self.current_state = self.close_look_around

    def parse_camera_data(self):
        new_detec = []
        for i in range(0,5):
            pc = self.turtle.get_point_cloud()
            image = self.turtle.get_rgb_image()
            imgcpy=image.copy()
            self.new_detected_cones = DetectedCones(self.turtle)  # -> detectedCones.red, green, blue
            if not self.new_detected_cones.parse_camera_data(image, pc):
                break
            self.new_detected_cones.draw_cones(
                imgcpy)  # -> az na konec, prekresli puvodni obrazek mohlo by se s nim pak hure pracovat
            new_detec+=self.new_detected_cones.all
        self.new_detected_cones = DetectedCones(self.turtle)
        for c in self.merge_multiple_detections(new_detec):
            self.new_detected_cones.add_cone(c)
        self.merge_new_cones()

    def merge_new_cones(self):
        for new_cone in self.new_detected_cones.all:
            if new_cone not in self.detected_cones.all:
                self.detected_cones.add_cone(new_cone)
            else:
                for i in range(0,len(self.detected_cones.all)):
                    if self.detected_cones.all[i] == new_cone:
                        if self.detected_cones.all[i].distance > new_cone.distance:
                            self.detected_cones.all[i] = new_cone


    def merge_multiple_detections(self,detections):
        similar = []
        indexpop = []
        for i in range(0,len(detections)):
            sim = []
            for j in range(0,len(detections)):
                if j not in indexpop and detections[i]==detections[j]:
                    sim.append(detections[j])
                    indexpop.append(j)
            if len(sim)>0:
                similar.append(sim)
        out = []
        for sim in similar:
            distances = []
            for cone in sim:
                distances.append(cone.distance)
            p = np.median(distances)
            for cone in sim:
                if cone.distance == p:
                    out.append(cone)
                    break
        return out

    def calculate_move(self):
        pair = self.detected_cones.get_closest_pair(self.last_cone_color)
        if pair is not None:
            self.actual_cone_color = pair[0].color
            first = (pair[0].distance * np.sin(pair[0].turtle_rotation - pair[0].angle_for_rotation),
                     pair[0].distance * np.cos(pair[0].turtle_rotation - pair[0].angle_for_rotation))
            second = (pair[1].distance * np.sin(pair[1].turtle_rotation - pair[1].angle_for_rotation),
                      pair[1].distance * np.cos(pair[1].turtle_rotation - pair[1].angle_for_rotation))
            self.center = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
            if self.ready_to_drive_through:
                dist1 = np.sqrt(self.center[0] ** 2 + self.center[1] ** 2)
                self.angle = np.arcsin(self.center[0] / dist1)
                self.distance = dist1
                self.distance -= 0.1
                self.angle_to_turn = -self.turtle.get_odometry()[2]+self.angle
                self.current_state = self.turn_to_calculated_angle
                return
            scale = MIDDLE_DIST_PRESET / ((second[1] - first[1]) ** 2 + (first[0] - second[0]) ** 2)
            goal1 = (self.center[0] + ((second[1] - first[1]) / 2) * scale,
                     self.center[1] + ((first[0] - second[0]) / 2) * scale)
            goal2 = (self.center[0] - ((second[1] - first[1]) / 2) * scale,
                     self.center[1] - ((first[0] - second[0]) / 2) * scale)
            #print("goal1", goal1, "goal2", goal2)
            dist1 = np.sqrt(goal1[0] ** 2 + goal1[1] ** 2)
            dist2 = np.sqrt(goal2[0] ** 2 + goal2[1] ** 2)
            if dist1 < dist2:
                self.angle = np.arcsin(goal1[0] / dist1)
                self.distance = dist1

            else:
                self.angle = -np.arcsin(goal2[0] / dist2)
                self.angle = -self.angle
                self.distance = dist2
            self.alpha = np.arcsin(self.center[0] / np.sqrt(self.center[0] ** 2 + self.center[1] ** 2))
            self.distance -= 0.1
            self.distance -= self.distance * 0.06
            self.angle_to_turn = -self.turtle.get_odometry()[2]+self.angle
            self.current_state = self.turn_to_calculated_angle


    def turn_to_calculated_angle(self):
        if self.angle_to_turn > 0:
            if self.turn_to_desired_angle(self.angle_to_turn,"left",0.3):
                self.turtle.reset_odometry()
                if self.ready_to_drive_through:
                    self.current_state = self.drive_through
                else:
                    self.current_state = self.prepare_for_drive_through
        else:
            if self.turn_to_desired_angle(abs(self.angle_to_turn),"right",0.3):
                self.turtle.reset_odometry()
                if self.ready_to_drive_through:
                    self.current_state = self.drive_through
                else:
                    self.current_state = self.prepare_for_drive_through

    def prepare_for_drive_through(self):
        odom = self.turtle.get_odometry()
        print("dist",np.sqrt(odom[0] ** 2 + odom[1] ** 2),self.distance)
        if np.sqrt(odom[0] ** 2 + odom[1] ** 2) < self.distance:
            self.turtle.cmd_velocity(linear=0.3, angular=0)
        else:
            self.ready_to_drive_through = True
            self.turtle.reset_odometry()
            self.detected_cones = DetectedCones(self.turtle)  # throw out all detected cones
            self.counter = 1
            self.direction = -1 if self.angle - self.alpha > 0 else 1
            self.current_state = self.turn_to_initial_angle

    def drive_through(self):
        if not self.last_cone_color == Color.INVALID and self.actual_cone_color == Color.GREEN: #already went through some cones
            self.finish = True
        if not self.actual_cone_color == Color.INVALID:
            self.last_cone_color = self.actual_cone_color
        self.actual_cone_color = Color.INVALID
        odom = self.turtle.get_odometry()
        ds = self.distance
        if self.finish:
            ds = self.distance + 0.4
        if np.sqrt(odom[0] ** 2 + odom[1] ** 2) < ds:
            self.turtle.cmd_velocity(linear=0.3, angular=0)
        else:
            if self.finish:
                self.current_state = self.fun
                return
            self.ready_to_drive_through = True
            self.direction = 2
            self.detected_cones = DetectedCones(self.turtle)
            self.counter = 1
            self.turtle.reset_odometry()
            self.current_state = self.turn_to_initial_angle

    def fun(self):
        if self.bumper_error:
            self.turtle.cmd_velocity(linear=0, angular=0)
        else:
            self.fun_step += 1
            self.fun_step %= 7
            self.turtle.play_sound(self.fun_step)
            self.turtle.cmd_velocity(linear=0, angular=1)

    # stop robot
    def bumper_cb(self,msg):
        self.current_state = self.fun
        self.bumper_error = True

def main():
    global stop
    turtle = Turtlebot(pc=True, rgb=True, depth=True)
    turtle.reset_odometry()
    cv2.namedWindow(WINDOW)  # display rgb image
    state_machine = StateMachine(turtle)
    plt.ion()
    turtle.register_bumper_event_cb(state_machine.bumper_cb)
    while not turtle.is_shutting_down():
        state_machine.run_state()
        if state_machine.detected_cones is not None:
            plt.clf()
            for cone in state_machine.detected_cones.all:
                if cone.color is Color.RED:
                    plt.scatter(cone.turtle_rotation - cone.angle_for_rotation, cone.distance, s=40, color='red')
                if cone.color is Color.GREEN:
                    plt.scatter(cone.turtle_rotation - cone.angle_for_rotation, cone.distance, s=40, color='green')
                if cone.color is Color.BLUE:
                    plt.scatter(cone.turtle_rotation - cone.angle_for_rotation, cone.distance, s=40, color='blue')
        plt.pause(0.00001)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
