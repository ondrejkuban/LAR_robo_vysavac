from cone import ColorsThresholds, Cone, get_color_for_threshold, get_threshold_for_color,Color
import cv2
import numpy as np

SURFACE_THRESHOLD = 400


class DetectedCones:
    def __init__(self, turtle):
        self.red = []
        self.green = []
        self.blue = []
        self.turtle = turtle
        self.all = []
        self.mask = None

    def detect_cones(self, image, point_cloud):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.red, mask_r = get_cones_for_color(hsv, ColorsThresholds.RED, self.turtle)
        self.green, mask_g = get_cones_for_color(hsv, ColorsThresholds.GREEN, self.turtle)
        self.blue, mask_b = get_cones_for_color(hsv, ColorsThresholds.BLUE, self.turtle)
        get_distances_for_cones(point_cloud, self.red, mask_r)
        get_distances_for_cones(point_cloud, self.green, mask_g)
        get_distances_for_cones(point_cloud, self.blue, mask_b)
        self.mask = mask_b
        for cone in self.red:
            self.all.append(cone)
        for cone in self.green:
            self.all.append(cone)
        for cone in self.blue:
            self.all.append(cone)
        self.red.sort(key=lambda cone: cone.distance)  # bude fungovat??? (dostanu cone a sort podle jeji distance)
        self.green.sort(key=lambda cone: cone.distance)  # bude fungovat???
        self.blue.sort(key=lambda cone: cone.distance)  # bude fungovat???

    def add_cone(self,cone):
        self.all.append(cone)
        if cone.color == Color.RED:
            self.red.append(cone)
        elif cone.color == Color.GREEN:
            self.green.append(cone)
        elif cone.color == Color.BLUE:
            self.blue.append(cone)
        self.red.sort(key=lambda cone: cone.distance)  # bude fungovat??? (dostanu cone a sort podle jeji distance)
        self.green.sort(key=lambda cone: cone.distance)  # bude fungovat???
        self.blue.sort(key=lambda cone: cone.distance)

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
        # chci navratit nejblizsi dvojici
        # pokud neni dvojice vrat nejblizsi
        # pokud neni nejblizsi vrat None


def detection_is_valid(detection):
    if detection[4] < SURFACE_THRESHOLD:
        return False
    if detection[2] * 2.5 > detection[3]:
        return False
    return True


def get_cones_for_color(image, threshold: tuple, turtle):
    mask = cv2.inRange(image, threshold[0], threshold[1])
    detections = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    results = []
    for i in range(1, detections[0]):
        if detection_is_valid(detections[2][i]):
            cone = Cone(get_color_for_threshold(threshold),
                        (detections[2][i][0], detections[2][i][1]),
                        (detections[2][i][2], detections[2][i][3]))
            cone.odo = turtle.get_odometry()[2]
            if cone.pt1[0] < 10 or cone.pt2[0] > 630 or cone.pt2[1]>240:
                pass
            else:
                results.append(cone)

    return results, mask


def draw_rectangles(image, cones: list):
    for cone in cones:
        cv2.rectangle(image, cone.pt1, cone.pt2, color=get_threshold_for_color(cone.color), thickness=2)
        cv2.putText(image, str(round(cone.distance, 2)), cone.pt1, cv2.FONT_ITALIC, 1,
                    get_threshold_for_color(cone.color), 2)


def calculate_euclidean(first_point):  # points[2] for x and points[0] for y
    return np.sqrt((first_point[0]) ** 2 + (first_point[1]) ** 2)


def get_distances_for_cones(point_cloud, cones, mask):
    for cone in cones:
        cone.x = get_point_in_space(point_cloud, cone, 2, mask)
        cone.y = get_point_in_space(point_cloud, cone, 0, mask)
        cone.distance = calculate_euclidean((cone.x, cone.y))
        cone.angle = np.arcsin(cone.y / cone.distance)


def get_point_in_space(point_cloud, cone, axis, mask):
    points = []
    for i in range(cone.pt1[0], cone.pt2[0]):
        for j in range(cone.pt1[1], cone.pt2[1]):
            if mask[j][i] == 255:
                if not np.isnan(point_cloud[j][i][axis]):
                    points.append(point_cloud[j][i][axis])
    return round(np.median(points), 3)
