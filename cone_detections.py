from cone import ColorsThresholds, Cone, get_color_for_threshold, get_threshold_for_color, Color
import cv2
import numpy as np

SURFACE_THRESHOLD = 800
MASK_THRESHOLD = 300


class DetectedCones:
    def __init__(self, turtle):
        self.red = []
        self.green = []
        self.blue = []
        self.all = []
        self.turtle = turtle

    def detect_cones(self, image, point_cloud):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.red, mask_r = get_cones_for_color(hsv, ColorsThresholds.RED, self.turtle)
        self.green, mask_g = get_cones_for_color(hsv, ColorsThresholds.GREEN, self.turtle)
        self.blue, mask_b = get_cones_for_color(hsv, ColorsThresholds.BLUE, self.turtle)
        if len(self.red) == 0 and len(self.blue) == 0 and len(self.green) == 0:
            return False
        get_distances_for_cones(point_cloud, self.red, mask_r)
        get_distances_for_cones(point_cloud, self.green, mask_g)
        get_distances_for_cones(point_cloud, self.blue, mask_b)
        self.all = [*self.red, *self.green, *self.blue]
        self.red.sort(key=lambda cone: cone.distance)
        self.green.sort(key=lambda cone: cone.distance)
        self.blue.sort(key=lambda cone: cone.distance)
        return True

    def add_cone(self, cone):
        self.all.append(cone)
        if cone.color == Color.RED:
            self.red.append(cone)
            self.red.sort(key=lambda c: c.distance)
        elif cone.color == Color.GREEN:
            self.green.append(cone)
            self.green.sort(key=lambda c: c.distance)
        elif cone.color == Color.BLUE:
            self.blue.append(cone)
            self.blue.sort(key=lambda c: c.distance)

    def draw_cones(self, image):
        draw_contours(image, self.red)
        draw_contours(image, self.green)
        draw_contours(image, self.blue)

    def get_search_color(self, last_color):
        search_color = []
        if last_color == Color.INVALID:  # Search for green
            search_color.append(Color.GREEN)
        elif last_color == Color.RED:  # Search for blue or green
            search_color.append(Color.BLUE)
            search_color.append(Color.GREEN)
        elif last_color == Color.BLUE:  # Search for red or green
            search_color.append(Color.GREEN)
            search_color.append(Color.RED)
        elif last_color == Color.GREEN:  # Search for Blue or red
            search_color.append(Color.BLUE)
            search_color.append(Color.RED)
        return search_color

    def get_distance_between_cones(self, first, second):
        first_true_angle = first.angle_for_rotation - first.turtle_rotation
        second_true_angle = second.angle_for_rotation - second.turtle_rotation
        cones_distance = np.sqrt(
            first.distance ** 2 + second.distance ** 2 - 2 * first.distance * second.distance * np.cos(
                first_true_angle - second_true_angle))
        return cones_distance

    def get_closest_pair(self, last_color):
        search_color = self.get_search_color(last_color)
        searched_cones = []

        if len(self.red) > 1 and Color.RED in search_color:
            for cone in self.red:
                searched_cones.append(cone)
        if len(self.green) > 1 and Color.GREEN in search_color:
            for cone in self.green:
                searched_cones.append(cone)
        if len(self.blue) > 1 and Color.BLUE in search_color:
            for cone in self.blue:
                searched_cones.append(cone)
        closest_cone = Cone(Color.INVALID, None, (-100, -100), None, None)
        if len(searched_cones) > 0:
            closest_cone = min(searched_cones, key=lambda c: c.distance)
        else:
            return None

        if closest_cone.color == Color.RED and len(self.red) > 1:  # red
            first = self.red[0]
            for i in range(1, len(self.red)):
                second = self.red[i]
                if self.get_distance_between_cones(first, second) > 0.20:
                    return [first, second]
        elif closest_cone.color == Color.GREEN and len(self.green) > 1:  # green
            first = self.green[0]
            for i in range(1, len(self.green)):
                second = self.green[i]
                if self.get_distance_between_cones(first, second) > 0.20:
                    return [first, second]
        elif closest_cone.color == Color.BLUE and len(self.blue) > 1:  # blue
            first = self.blue[0]
            for i in range(1, len(self.blue)):
                second = self.blue[i]
                if self.get_distance_between_cones(first, second) > 0.20:
                    return [first, second]
        return None


def detection_is_valid(detection):
    if detection[2] * 2.5 > detection[3]:
        return False
    if detection[3] < 30:
        return False
    return True


def is_mask_valid(mask):
    return sum(map(sum, mask)) // 255 > MASK_THRESHOLD


def get_cones_for_color(image, threshold: tuple, turtle):
    mask = cv2.inRange(image, threshold[0], threshold[1])
    if is_mask_valid(mask):
        detections, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for contour in detections:
            bound_box = cv2.boundingRect(contour)
            if cv2.contourArea(contour) >= SURFACE_THRESHOLD and detection_is_valid(bound_box):
                moment = cv2.moments(contour)
                cX = int(moment["m10"] / moment["m00"])
                cY = int(moment["m01"] / moment["m00"])
                cone = Cone(get_color_for_threshold(threshold), contour, (cX, cY), bound_box, turtle.get_odometry()[2])
                results.append(cone)
        return results, mask
    return [], []


def draw_contours(image, cones: list):
    for cone in cones:
        cv2.drawContours(image, [cone.contour], -1, get_threshold_for_color(cone.color), 2)
        cv2.putText(image, str(round(cone.distance, 2)), cone.center, cv2.FONT_ITALIC, 1,
                    get_threshold_for_color(cone.color), 2)


def calculate_euclidean(first_point):
    return np.sqrt((first_point[0]) ** 2 + (first_point[1]) ** 2)


def get_distances_for_cones(point_cloud, cones, mask):
    for cone in cones:
        cone.x = get_point_in_space(point_cloud, cone, 2, mask)
        cone.y = get_point_in_space(point_cloud, cone, 0, mask)
        cone.distance = calculate_euclidean((cone.x, cone.y))
        cone.angle_for_rotation = np.arcsin(cone.y / cone.distance)


def get_point_in_space(point_cloud, cone, axis, mask):
    points = []
    start_width = max(int(cone.center[0] - cone.bounding_box[2] // 2.5), 0)
    end_width = min(int(cone.center[0] + cone.bounding_box[2] // 2.5), 640)
    start_height = max(int(cone.center[1] - cone.bounding_box[3] // 4), 0)
    end_height = min(int(cone.center[1] + cone.bounding_box[3] // 4), 480)

    for i in range(start_width, end_width):
        for j in range(start_height, end_height):
            if not np.isnan(point_cloud[j][i][axis]):
                if mask[j][i] == 255:
                    points.append(point_cloud[j][i][axis])

    return np.median(points)
