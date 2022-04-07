import cv2
from scipy.io import loadmat
import numpy as np
import skimage.exposure

matlab_data = [loadmat('2022-03-03-15-30-32.mat'),
               loadmat('2022-03-03-15-31-04.mat'),
               loadmat('2022-03-03-15-31-31.mat'),
               loadmat('2022-03-03-15-31-43.mat'),
               loadmat('2022-03-03-15-32-07.mat'),
               loadmat('2022-03-03-15-32-23.mat')]

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
        self.center = (position[0]+size[0]//2,position[1]+size[1]//2)


def detection_is_valid(detection):
    if detection[4] < surface_threshold:
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
        cv2.putText(image,str(cone.distance),cone.pt1,cv2.FONT_ITALIC,1,get_threshold_for_color(cone.color),2)


def calculate_euclidean(points):
    return round(np.sqrt(points[0]**2 + points[1]**2), 3)

# init
rgb_image = matlab_data[0]['image_rgb']
depth_image = matlab_data[0]['image_depth']
k_depth = matlab_data[0]['K_depth']
k_rgb = matlab_data[0]['K_rgb']
point_cloud = matlab_data[0]['point_cloud']
cv2.namedWindow("RGB")
cv2.namedWindow("DEPTH")
#print(rgb_image[481][420])
while True:
    #M = k_depth @ np.linalg.inv(k_rgb)
   # warped_rgb = cv2.warpPerspective(rgb_image, M, (640, 480))
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    get_cones_for_color(hsv, ColorsThresholds.BLUE)
    # creating mask to find rectangles
    red_cones = get_cones_for_color(hsv, ColorsThresholds.RED)
    green_cones = get_cones_for_color(hsv, ColorsThresholds.GREEN)
    blue_cones = get_cones_for_color(hsv, ColorsThresholds.BLUE)

   # for cone in red_cones:
    #    cone.distance = calculate_euclidean(point_cloud[cone.center[1]][cone.center[0]])
    #for cone in green_cones:
   #     cone.distance = calculate_euclidean(point_cloud[cone.center[1]][cone.center[0]])
   # for cone in blue_cones:
     #   cone.distance = calculate_euclidean(point_cloud[cone.center[1]][cone.center[0]])

    # drawing rectangle
    im = rgb_image.copy()
    draw_rectangles(im, red_cones)
    draw_rectangles(im, green_cones)
    draw_rectangles(im, blue_cones)

    minmax = cv2.minMaxLoc(depth_image)
    max = np.ceil(minmax[1])
    out = cv2.convertScaleAbs(depth_image, alpha=255 / max)
    draw_rectangles(out, red_cones)
    draw_rectangles(out, green_cones)
    draw_rectangles(out, blue_cones)

    cv2.imshow("RGB", im)
    cv2.imshow("DEPTH", out)
    cv2.waitKey(1)
