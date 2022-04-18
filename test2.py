

import cv2
from scipy.io import loadmat
import numpy as np
import imutils

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
    RED = ((0, 116, 70), (6.5, 255, 255))
    GREEN = ((28, 55, 45), (85, 255, 230))
    BLUE = ((90, 172, 42), (106, 255, 235))


class Cone:
    def __init__(self, color: int, position: tuple, size: tuple,contours,mask):
        self.color = color
        self.pt1 = position
        self.pt2 = (position[0] + size[0], position[1] + size[1])
        self.contours = contours
        self.size = size
        self.center = (position[0] + size[0] // 2, position[1] + size[1] // 2)
        self.mask = mask


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

    detec,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #detec = imutils.grab_contours(detec)
    #detections = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    results = []
    c_max = None
    for d in detec:
        if cv2.contourArea(d)>=200:
            results.append(Cone(get_color_for_threshold(threshold),
                                (0, 0),
                                (0, 0), d,mask))

    #for i in range(1, detections[0]):
       # if detection_is_valid(detections[2][i]):
      #      results.append(Cone(get_color_for_threshold(threshold),
       #                         (detections[2][i][0], detections[2][i][1]),
      #                          (detections[2][i][2], detections[2][i][3]),detec))

    return results

i=0
pc = matlab_data[i]['point_cloud']

def draw_rectangles(image, cones: list):
    for cone in cones:
        #for c in cone.contours:
        cv2.drawContours(image,[cone.contours],-1,get_threshold_for_color(cone.color),1)
        M = cv2.moments(cone.contours)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        #.circle(image, (cX, cY), 7,get_threshold_for_color(cone.color), -1)
        points = []
        box = cv2.boundingRect(cone.contours)
        for i in range(box[0],box[0]+box[2]):
            for j in range(box[1],box[1]+box[3]):
                if cone.mask[j][i] == 255:
                    t = True
                    cv2.circle(image, (i, j), 1, (255,255,255), -1)
                    for p in pc[j][i]:
                        if np.isnan(p):
                            t = False
                    if t:
                        dist = np.sqrt(pc[j][i][0]**2+pc[j][i][2]**2)
                        points.append([i,j,dist])
                        #cv2.circle(image, (i, j), 1, get_threshold_for_color(cone.color), -1)

        n = min(points,key=lambda x:x[2])[2]
        m = max(points, key=lambda l: l[2])
        m[2]-=n
        for p in points:
            s= (p[2]-n)/m[2]
            s = int(s*255)
            z = None
            if cone.color == Color.BLUE:
                z=(s,0,0)
            elif cone.color == Color.GREEN:
                z = (0, s, 0)
            elif cone.color == Color.RED:
                z = (0, 0, s)
            cv2.circle(image, (p[0],p[1]), 1,z, -1)
        #cv2.rectangle(image, cone.pt1, cone.pt2, color=get_threshold_for_color(cone.color), thickness=2)
        #cv2.putText(image, str(cone.distance), cone.pt1, cv2.FONT_ITALIC, 1, get_threshold_for_color(cone.color), 2)


def calculate_euclidean(points):
    return round(np.sqrt(points[0] ** 2 + points[1] ** 2), 3)





def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY

    if event == cv2.EVENT_LBUTTONDOWN:
        print(pc[y][x])
        mouseX,mouseY = x,y

# init
rgb_image = matlab_data[0]['image_rgb']
depth_image = matlab_data[0]['image_depth']
k_depth = matlab_data[0]['K_depth']
k_rgb = matlab_data[0]['K_rgb']
point_cloud = matlab_data[0]['point_cloud']
cv2.namedWindow("RGB")
#cv2.namedWindow("DEPTH")
# print(rgb_image[481][420])

while True:
    # M = k_depth @ np.linalg.inv(k_rgb)
    # warped_rgb = cv2.warpPerspective(rgb_image, M, (640, 480))
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    get_cones_for_color(hsv, ColorsThresholds.BLUE)
    # creating mask to find rectangles
    red_cones = get_cones_for_color(hsv, ColorsThresholds.RED)
    green_cones = get_cones_for_color(hsv, ColorsThresholds.GREEN)
    blue_cones = get_cones_for_color(hsv, ColorsThresholds.BLUE)

    for cone in red_cones:
        cone.distance = calculate_euclidean(point_cloud[cone.center[1]][cone.center[0]])
    for cone in green_cones:
        cone.distance = calculate_euclidean(point_cloud[cone.center[1]][cone.center[0]])
    for cone in blue_cones:
        cone.distance = calculate_euclidean(point_cloud[cone.center[1]][cone.center[0]])

    # drawing rectangle
    im = rgb_image.copy()
    draw_rectangles(im, red_cones)
    draw_rectangles(im, green_cones)
    draw_rectangles(im, blue_cones)


    cv2.imshow("RGB", im)
    #cv2.imshow("DEPTH", out)
    cv2.setMouseCallback('RGB', draw_circle)
    k = cv2.waitKey(1)
    if k==ord('f'):
        i+=1
        rgb_image = matlab_data[i]['image_rgb']
        depth_image = matlab_data[i]['image_depth']
        k_depth = matlab_data[i]['K_depth']
        k_rgb = matlab_data[i]['K_rgb']
        point_cloud = matlab_data[i]['point_cloud']

