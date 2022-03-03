from robolab_turtlebot import Turtlebot, detector
import numpy as np
import cv2

x_range = (-0.3, 0.3)
z_range = (0.3, 3.0)

WINDOW_D = 'obstacles' #depth
WINDOW = 'markers'

stop = False

def bumper_callBack():
    #stop robot
    state = 'STOP'
    stop = True

def main():

    turtle = Turtlebot(pc=True, rgb = True, depth = True)
    cv2.namedWindow(WINDOW_D)   #display depth
    cv2.namedWindow(WINDOW)     #display rgb image

    while not turtle.is_shutting_down() and not stop:
        # get point cloud
        pc = turtle.get_point_cloud()
        rgb = turtle.get_rgb_image()

        if (pc is None) or (rgb is None):
            continue

        # mask out floor points
        mask = pc[:, :, 1] > x_range[0]

        # mask point too far and close
        mask = np.logical_and(mask, pc[:, :, 2] > z_range[0])
        mask = np.logical_and(mask, pc[:, :, 2] < z_range[1])

        if np.count_nonzero(mask) <= 0:
            continue

        # empty image
        image = np.zeros(mask.shape)

        # assign depth i.e. distance to image
        image[mask] = np.int8(pc[:, :, 2][mask] / 3.0 * 255)
        im_color = cv2.applyColorMap(255 - image.astype(np.uint8),
                                     cv2.COLORMAP_JET)

        # draw markers in the image
        markers = detector.detect_markers(rgb)

        #draw markers in the image
        detector.draw_markers(rgb, markers)

        #Callback of bumper activation
        turtle.register_bumper_event_cb(bumper_callBack)

        # show image
        cv2.imshow(WINDOW_D, im_color)
        cv2.imshow(WINDOW,rgb)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
