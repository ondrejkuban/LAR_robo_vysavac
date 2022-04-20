from cone import ColorsThresholds, Cone, get_color_for_threshold, get_threshold_for_color, Color
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
        hsvs = []
        for im in image:
            hsvs.append(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        self.red, mask_r = get_cones_for_color(hsvs, ColorsThresholds.RED, self.turtle)
        self.green, mask_g = get_cones_for_color(hsvs, ColorsThresholds.GREEN, self.turtle)
        self.blue, mask_b = get_cones_for_color(hsvs, ColorsThresholds.BLUE, self.turtle)
        if len(self.red) == 0 and len(self.blue) == 0 and len(self.green) == 0:
            return False
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

    def add_cone(self, cone):
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

    def get_closest_pair(self, last_color):
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
        all_cones = []
        if len(self.red) > 1 and Color.RED in search_color:
            for cone in self.red:
                all_cones.append(cone)
        if len(self.green) > 1 and Color.GREEN in search_color:
            for cone in self.green:
                all_cones.append(cone)
        if len(self.blue) > 1 and Color.BLUE in search_color:
            for cone in self.blue:
                all_cones.append(cone)
        closest_cone = Cone(Color.INVALID, (-100, -100), (-100, -100))
        if len(all_cones) > 0:
            closest_cone = min(all_cones,key=lambda c: c.distance)
        else:
            return None

        if closest_cone.color == Color.RED and len(self.red) > 1:  # red
            first = self.red[0]
            for i in range(0,len(self.red)-1):
                
                second = self.red[i+1]
                cones_distance = np.sqrt(first.distance**2+second.distance**2-2*first.distance*second.distance*np.cos(first.angle-second.angle))
                if cones_distance < 0.35:
                    return [first, second]
        elif closest_cone.color == Color.GREEN and len(self.green) > 1:  # green
            first = self.green[0]
            for i in range(0,len(self.green)-1):
                second = self.green[i+1]
                cones_distance = np.sqrt(first.distance**2+second.distance**2-2*first.distance*second.distance*np.cos(first.angle-second.angle))
                if cones_distance < 0.35:
                    return [first, second]
        elif closest_cone.color == Color.BLUE and len(self.blue) > 1:  # blue
            first = self.blue[0]
            for i in range(0,len(self.blue)-1):
                second = self.blue[i+1]
                cones_distance = np.sqrt(first.distance**2+second.distance**2-2*first.distance*second.distance*np.cos(first.angle-second.angle))
                if cones_distance < 0.35:
                    return [first, second]
        return None
        # chci navratit nejblizsi dvojici
        # pokud neni dvojice vrat nejblizsi
        # pokud neni nejblizsi vrat None


def detection_is_valid(detection):
    if detection[4] < SURFACE_THRESHOLD:
        return False
    if detection[2] * 2.5 > detection[3]:
        return False
    if detection[3] < 30:
        return False
    # if detection[2]<30:
    #  return False
    return True


def is_mask_valid(mask):
    return sum(map(sum, mask)) // 255 > 300


def get_cones_for_color(image, threshold: tuple, turtle):
    masks = []
    detections_s = []
    for im in image:
        mask = cv2.inRange(im, threshold[0], threshold[1])
        if not is_mask_valid(mask):
            return [], []
        masks.append(mask)
        #if len(masks) != 2:
           # continue
        detections = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        detec,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for d in detec:
            if cv2.contourArea(d)>=800:
                cone = Cone(get_color_for_threshold(threshold),(0,0),(0,0))
                cone.odo = turtle.get_odometry()[2]
                cone.contour = d
                results.append(cone)
        '''        
        for i in range(1, detections[0]):
            if detection_is_valid(detections[2][i]):
                cone = Cone(get_color_for_threshold(threshold),
                            (detections[2][i][0], detections[2][i][1]),
                            (detections[2][i][2], detections[2][i][3]))
                cone.odo = turtle.get_odometry()[2]
                
                if cone.pt1[0] < 10 or cone.pt2[0] > 630 or cone.pt2[1] < 120:
                    pass
                else:
                    results.append(cone)
        '''
        detections_s.append(results)

    return detections_s[0], masks


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
    width = int(cone.pt2[0] - cone.pt1[0])
    center_w = int(cone.pt2[0] - width / 2)
    height = int(cone.pt2[1] - cone.pt1[1])
    center_h = int(cone.pt2[1] - height / 2)
    box = cv2.boundingRect(cone.contour)
    M = cv2.moments(cone.contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    stw = int(cX-box[2]//2.5)
    enw = int(cX+box[2]//2.5)
    sth = int(cY-box[3]//4)
    enh = int(cY+box[3]//4)
    for p in range(0, len(point_cloud)):
        for i in range(max(stw,0),min(enw,640)):
            for j in range(max(sth,0),min(enh,480)):
                if not np.isnan(point_cloud[p][j][i][axis]):
                    if mask[p][j][i] == 255:
                        points.append(point_cloud[p][j][i][axis])
    # if axis == 2 and len(points)>0:
    ##print("points",points[0],points[-1])

    return np.median(points)
