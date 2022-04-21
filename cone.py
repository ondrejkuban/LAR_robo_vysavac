# color class
angle_threshold = 0.08
distance_threshold = 0.5
class Color:
    INVALID = 0
    RED = 1
    GREEN = 2
    BLUE = 3


class ColorsThresholds:
    #       dark           light
    RED = ((0, 116, 90), (6.5, 255, 255))
    GREEN = ((28, 55, 45), (85, 255, 230))
    BLUE = ((90, 172, 42), (106, 255, 235))


class Cone:
    def __init__(self, color: int, contour, center: tuple, bounding_box,turtle_rotation):
        self.color = color
        self.center = center
        self.contour = contour
        self.bounding_box = bounding_box
        self.turtle_rotation = turtle_rotation
        self.x = None
        self.y = None
        self.distance = None
        self.angle_for_rotation = None

    def __eq__(self, other):
        this_true_angle = self.angle_for_rotation - self.turtle_rotation
        other_true_angle = other.angle_for_rotation - other.turtle_rotation
        angle_similar = this_true_angle - angle_threshold < other_true_angle < this_true_angle + angle_threshold
        return angle_similar and self.color == other.color and abs(self.distance - other.distance) < distance_threshold


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
