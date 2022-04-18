# color class
class Color:
    INVALID = 0
    RED = 1
    GREEN = 2
    BLUE = 3


class ColorsThresholds:
    #       dark           light
    RED = ((0, 116, 90), (6.5, 255, 255))
    GREEN = ((34, 55, 45), (65, 255, 230))
    BLUE = ((90, 172, 42), (106, 255, 235))



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
        self.odo = None

    def __eq__(self, other):
        if (self.angle - self.odo - 0.05 < other.angle - other.odo < self.angle - self.odo + 0.05) and self.color==other.color:
            self.distance = min(self.distance,other.distance)
        return ((self.angle - self.odo - 0.05 < other.angle - other.odo < self.angle - self.odo + 0.05) and \
               abs(self.distance - other.distance) < 50)


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
