from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate
from robolab_turtlebot import detector
import numpy as np
import cv2
import time

x_range = (-0.3, 0.3)
z_range = (0.3, 3.0)

WINDOW_D = 'DEPTH'  # depth
WINDOW = 'RGB'
SURFACE_THRESHOLD = 400

stop = False
fun_step = 0


# color class
class Color:
    INVALID = 0
    RED = 1
    GREEN = 2
    BLUE = 3


class ColorsThresholds:
    #       dark           light
    RED = ((0, 100, 50), (5, 255, 255))
    GREEN = ((45, 70, 20), (75, 255, 255))
    BLUE = ((90, 100, 20), (110, 255, 255))


class Cone:
    def __init__(self, color: int, position: tuple, size: tuple):
        self.color = color
        self.pt1 = position
        self.pt2 = (position[0] + size[0], position[1] + size[1])
        self.size = size
        self.center = (position[0] + size[0] // 2, position[1] + size[1] // 2)
        self.distance = -1
        self.x = None
        self.y = None
        self.angle = None


class PID:
    def __init__(self):
        self.p_gain = 1.25
        self.i_gain = 1
        self.d_gain = 1
        self.goal = 0

    def get_new_output(self, measurement):
        if self.p_gain * (measurement - self.goal) > 1:
            return 1
        if self.p_gain * (measurement - self.goal) < -1:
            return -1
        if -0.1 < self.p_gain * (measurement - self.goal) < 0:
            return -0.1
        if 0.1 > self.p_gain * (measurement - self.goal) > 0:
            return 0.1
        return self.p_gain * (measurement - self.goal)


class DetectedCones:
    def __init__(self):
        self.red = None
        self.green = None
        self.blue = None

    def detect_cones(self, image, point_cloud):
        self.red = get_cones_for_color(image, ColorsThresholds.RED)
        self.green = get_cones_for_color(image, ColorsThresholds.GREEN)
        self.blue = get_cones_for_color(image, ColorsThresholds.BLUE)
        get_distances_for_cones(point_cloud, self.red)
        get_distances_for_cones(point_cloud, self.green)
        get_distances_for_cones(point_cloud, self.blue)
        self.red.sort(key=lambda cone: cone.distance)  # bude fungovat??? (dostanu cone a sort podle jeji distance)
        self.green.sort(key=lambda cone: cone.distance)  # bude fungovat???
        self.blue.sort(key=lambda cone: cone.distance)  # bude fungovat???

    def draw_cones(self, image):
        draw_rectangles(image, self.red)
        draw_rectangles(image, self.green)
        draw_rectangles(image, self.blue)

    def get_closest_pair(self):
        closest_cone = None
        closest_cone = min(x for x in self if x is not None)  # moje duvera v tuhle radku je maximalne 5 (slovy pět)%
        if closest_cone.color == 1 and len(self.red > 1):  # red
            return [self.red[0], self.red[1]]
        elif closest_cone.color == 2 and len(self.green > 1):  # green
            return [self.green[0], self.green[1]]
        elif closest_cone.color == 3 and len(self.blue > 1):  # blue
            return [self.blue[0], self.blue[1]]
        return closest_cone
        ## chci navratit nejblizsi dvojici
        ## pokud neni dvojice vrat nejblizsi
        ## pokud neni nejblizsi vrat None


def detection_is_valid(detection):
    if detection[4] < SURFACE_THRESHOLD:
        return False
    if detection[2] * 2 > detection[3]:
        return False
    return True


def get_color_for_threshold(threshold):
    return {
        ColorsThresholds.RED: Color.RED,
        ColorsThresholds.GREEN: Color.GREEN,
        ColorsThresholds.BLUE: Color.BLUE
    }.get(threshold)


def get_threshold_for_color(color):
    return {
        Color.RED: (0, 0, 255),
        Color.GREEN: (0, 255, 0),
        Color.BLUE: (255, 0, 0)
    }.get(color)


# Return "array of Cones" from image
def get_cones_for_color(image, threshold: tuple):
    mask = cv2.inRange(image, threshold[0], threshold[1])
    detections = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    results = []
    for i in range(1, detections[0]):
        if detection_is_valid(detections[2][i]):
            results.append(Cone(get_color_for_threshold(threshold),
                                (detections[2][i][0], detections[2][i][1]),
                                (detections[2][i][2], detections[2][i][3])))

    return results


def draw_rectangles(image, cones: list):
    for cone in cones:
        cv2.rectangle(image, cone.pt1, cone.pt2, color=get_threshold_for_color(cone.color), thickness=2)
        cv2.putText(image, str(cone.distance), cone.pt1, cv2.FONT_ITALIC, 1, get_threshold_for_color(cone.color), 2)


def calculate_euclidean(first_point):  # points[2] for x and points[0] for y
    return np.sqrt((first_point[0]) ** 2 + (first_point[1]) ** 2)


def get_distances_for_cones(point_cloud, cones):
    for cone in cones:
        cone.x = get_point_in_space(point_cloud, cone, 2)
        cone.y = get_point_in_space(point_cloud, cone, 0)
        cone.distance = calculate_euclidean((cone.x, cone.y))
        cone.angle = np.arcsin(cone.y / cone.distance)


def get_point_in_space(point_cloud, cone, axis):
    points = []
    for i in range(cone.pt1[0], cone.pt2[0]):
        for j in range(cone.pt1[1], cone.pt2[1]):
            if not np.isnan(point_cloud[j][i][axis]):
                points.append(point_cloud[j][i][axis])
    return round(np.median(points), 3)


def fun(turtle):
    # global fun_step
    # fun_step += 1
    # fun_step %= 7
    # turtle.play_sound(fun_step)
    # turtle.cmd_velocity(linear=0, angular=0.5)
    time.sleep(0.4)


# stop robot
def bumper_callBack(msg):
    global stop
    stop = True
    print('Bumper was activated, new state is STOP')


def main():
    global stop
    turtle = Turtlebot(pc=True, rgb=True, depth=True)
    cv2.namedWindow(WINDOW)  # display rgb image
    turtle.register_bumper_event_cb(bumper_callBack)
    pid = PID()
    while not turtle.is_shutting_down():
        # get point cloud

        depth = turtle.get_depth_image()
        point_cloud = turtle.get_point_cloud()
        rgb = turtle.get_rgb_image()

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        im = rgb.copy()

        detectedCones = DetectedCones()  # -> detectedCones.red, green, blue
        detectedCones.detect_cones(hsv, point_cloud)
        detectedCones.draw_cones(im)  # -> az na konec, prekresli puvodni obrazek mohlo by se s nim pak hure pracovat

        # drawing rectangle

        minmax = cv2.minMaxLoc(depth)
        max = np.ceil(minmax[1])
        out = cv2.convertScaleAbs(depth, alpha=255 / max)
        detectedCones.draw_cones(out)
        pair = detectedCones.get_closest_pair()
        print(pair[0].distance,pair[1].distance)
        if not stop:
            if len(detectedCones.red) > 1 and detectedCones.red[0].angle is not None and detectedCones.red[
                1].angle is not None:
                error = (detectedCones.red[0].angle + detectedCones.red[1].angle) / 2
                if abs(error) > 0.09:
                    turtle.cmd_velocity(linear=0, angular=-pid.get_new_output(error))
                else:
                    turtle.cmd_velocity(linear=0.65, angular=0.0)
            elif len(detectedCones.red) > 0 and detectedCones.red[0].angle is not None:
                if detectedCones.red[0].center[0] > 320 and detectedCones.red[0].distance > 0.55:
                    turtle.cmd_velocity(linear=0.0, angular=-0.35)
                else:
                    turtle.cmd_velocity(linear=0.0, angular=0.35)
            else:  # pojede rovne pokud nic nenajde????
                turtle.cmd_velocity(linear=0.65, angular=0.0)
        else:
            fun(turtle)
        cv2.imshow("RGB", im)

        cv2.waitKey(1)


if __name__ == '__main__':
    main()
