from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate
from robolab_turtlebot import detector, get_time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import cv2
import time

x_range = (-0.3, 0.3)
z_range = (0.3, 3.0)

WINDOW_D = 'DEPTH'  # depth
WINDOW = 'RGB'
SURFACE_THRESHOLD = 400

stop = False
t = 0
fun_step = 0


# color class
class Color:
    INVALID = 0
    RED = 1
    GREEN = 2
    BLUE = 3


class ColorsThresholds:
    #       dark           light
    RED = [[0, 116, 114], [6.5, 255, 255]]
    GREEN = [[39, 74, 54], [80, 255, 230]]
    BLUE = [[90, 172, 42], [106, 255, 252]]


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
        self.maskr = None
        self.maskb = None
        self.maskg = None

    def detect_cones(self, image, point_cloud):
        self.red,self.maskr = get_cones_for_color(image, ColorsThresholds.RED)
        self.green,self.maskg = get_cones_for_color(image, ColorsThresholds.GREEN)
        self.blue,self.maskb = get_cones_for_color(image, ColorsThresholds.BLUE)
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
        all_cones = self.red + self.green + self.blue
        if len(all_cones) > 0:
            closest_cone = min(all_cones,
                               key=lambda cone: cone.distance)  # moje duvera v tuhle radku je maximalne 5 (slovy pět)%
        else:
            return []
        if closest_cone.color == 1 and len(self.red) > 1:  # red
            return [self.red[0], self.red[1]]
        elif closest_cone.color == 2 and len(self.green) > 1:  # green
            return [self.green[0], self.green[1]]
        elif closest_cone.color == 3 and len(self.blue) > 1:  # blue
            return [self.blue[0], self.blue[1]]
        return [closest_cone]
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
def get_cones_for_color(image, threshold):
    print(threshold[0])
    mask = cv2.inRange(image,  tuple(threshold[0]),  tuple(threshold[1]))
    detections = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    results = []
    #for i in range(1, detections[0]):
      #  if detection_is_valid(detections[2][i]):
        #    results.append(Cone(tuple(get_color_for_threshold(threshold)),
         #                       (detections[2][i][0], detections[2][i][1]),
         #                       (detections[2][i][2], detections[2][i][3])))

    return results,mask


def draw_rectangles(image, cones: list):
    pass
    for cone in cones:
        cv2.rectangle(image, cone.pt1, cone.pt2, color=tuple(get_threshold_for_color(cone.color)), thickness=2)
        cv2.putText(image, str(round(cone.distance, 2)), cone.pt1, cv2.FONT_ITALIC, 1,
                    tuple(get_threshold_for_color(cone.color)), 2)


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
    global t
    print(turtle.get_odometry())
    if abs(turtle.get_odometry()[2]) < np.pi:
        turtle.cmd_velocity(linear=0, angular=1)
    else:
        turtle.cmd_velocity(linear=0, angular=0)


# stop robot
def bumper_callBack(msg):
    global stop
    global t
    t = get_time()

    stop = True
    print('Bumper was activated, new state is STOP')


def update(val):
    print(val)

def updateHL(val):
    ColorsThresholds.BLUE[1][0] = val
    print(val)

def updateVL(val):
    ColorsThresholds.BLUE[1][2] = val
    print(val)

def updateSL(val):
    ColorsThresholds.BLUE[1][1] = val
    print(val)

def updateHD(val):
    ColorsThresholds.BLUE[0][0] = val
    print(val)

def updateVD(val):
    ColorsThresholds.BLUE[0][2] = val
    print(val)

def updateSD(val):
    ColorsThresholds.BLUE[0][1] = val
    print(val)

def main():
    global stop
    turtle = Turtlebot(pc=True, rgb=True, depth=True)
    state = 0
    cv2.namedWindow(WINDOW)  # display rgb image
    turtle.register_bumper_event_cb(bumper_callBack)
    pid = PID()
    angle = 0
    distance = 0
    turtle.reset_odometry()
    axfreq = plt.axes([.05, 0.25, 0.0225, 0.63])
    freq_slider = Slider(
        ax=axfreq,
        label='HL',
        valmin=0,
        valmax=255,
        valinit=ColorsThresholds.BLUE[1][0],
        orientation="vertical"
    )

    # Make a vertically oriented slider to control the amplitude
    axamp = plt.axes([0.15, 0.25, 0.0225, 0.63])
    amp_slider = Slider(
        ax=axamp,
        label="SL",
        valmin=0,
        valmax=255,
        valinit=ColorsThresholds.BLUE[1][1],
        orientation="vertical"
    )
    ayamp = plt.axes([0.25, 0.25, 0.0225, 0.63])
    p_slider = Slider(
        ax=ayamp,
        label="VL",
        valmin=0,
        valmax=255,
        valinit=ColorsThresholds.BLUE[1][2],
        orientation="vertical"
    )
    axfreq = plt.axes([0.5, 0.25, 0.0225, 0.63])
    freq_sliderD = Slider(
        ax=axfreq,
        label='HD',
        valmin=0,
        valmax=255,
        valinit=ColorsThresholds.BLUE[0][0],
        orientation="vertical"
    )

    # Make a vertically oriented slider to control the amplitude
    axamp = plt.axes([0.6, 0.25, 0.0225, 0.63])
    amp_sliderD= Slider(
        ax=axamp,
        label="SD",
        valmin=0,
        valmax=255,
        valinit=ColorsThresholds.BLUE[0][1],
        orientation="vertical"
    )
    ayamp = plt.axes([0.7, 0.25, 0.0225, 0.63])
    p_sliderD = Slider(
        ax=ayamp,
        label="VD",
        valmin=0,
        valmax=255,
        valinit=ColorsThresholds.BLUE[0][2],
        orientation="vertical"
    )
    freq_slider.on_changed(updateHL)
    amp_slider.on_changed(updateSL)
    p_slider.on_changed(updateVL)
    freq_sliderD.on_changed(updateHD)
    amp_sliderD.on_changed(updateSD)
    p_sliderD.on_changed(updateVD)
    while not turtle.is_shutting_down():
        # get point cloud
        depth = turtle.get_depth_image()
        point_cloud = turtle.get_point_cloud()
        rgb = turtle.get_rgb_image()

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        im = rgb.copy()

        detectedCones = DetectedCones()  # -> detectedCones.red, green, blue
        detectedCones.detect_cones(hsv, point_cloud)
        #detectedCones.draw_cones(im)  # -> az na konec, prekresli puvodni obrazek mohlo by se s nim pak hure pracovat

        # drawing rectangle

        minmax = cv2.minMaxLoc(depth)
        max = np.ceil(minmax[1])
        out = cv2.convertScaleAbs(depth, alpha=255 / max)
       # detectedCones.draw_cones(out)
        #print(detectedCones.maskb)
        cv2.imshow("RGB", detectedCones.maskb)
        plt.pause(0.001)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()