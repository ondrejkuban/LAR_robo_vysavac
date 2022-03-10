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

stop = False
fun_step = 0


def fun(turtle):
    global fun_step
    fun_step += 1
    fun_step %= 7
    turtle.play_sound(fun_step)
    time.sleep(0.4)
    turtle.cmd_velocity(angular=1)


# stop robot
def bumper_callBack(msg):
    global stop
    stop = True
    print('Bumper was activated, new state is STOP')


surface_threshold = 400


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


def detection_is_valid(detection):
    if detection[4] < surface_threshold:
        return False
    if detection[2] * 3 > detection[3]:
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


def main():
    global stop
    turtle = Turtlebot(pc=True, rgb=True, depth=True)
    #cv2.namedWindow(WINDOW_D)  # display depth
    cv2.namedWindow(WINDOW)  # display rgb image

    while not turtle.is_shutting_down():
        # get point cloud
        if not stop:
            turtle.cmd_velocity(linear=0.0)
        else:
            fun(turtle)
            turtle.cmd_velocity(linear=0)
       # pc = turtle.get_dep
        rgb = turtle.get_rgb_image()

        #M = k_depth @ np.linalg.inv(k_rgb)
        #warped_rgb = cv2.warpPerspective(rgb_image, M, (640, 480))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        get_cones_for_color(hsv, ColorsThresholds.BLUE)
        # creating mask to find rectangles
        red_cones = get_cones_for_color(hsv, ColorsThresholds.RED)
        green_cones = get_cones_for_color(hsv, ColorsThresholds.GREEN)
        blue_cones = get_cones_for_color(hsv, ColorsThresholds.BLUE)

        # drawing rectangle
        im = rgb.copy()
        draw_rectangles(im, red_cones)
        draw_rectangles(im, green_cones)
        draw_rectangles(im, blue_cones)

        #minmax = cv2.minMaxLoc(depth_image)
        #max = np.ceil(minmax[1])
        #out = cv2.convertScaleAbs(depth_image, alpha=255 / max)
        #draw_rectangles(out, red_cones)
        #draw_rectangles(out, green_cones)
        #draw_rectangles(out, blue_cones)

        cv2.imshow("RGB", im)
        #cv2.imshow("DEPTH", out)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
