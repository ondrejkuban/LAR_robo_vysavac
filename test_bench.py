
import cv2
import scipy.io
import numpy as np

mat = scipy.io.loadmat('2022-03-03-15-30-32.mat')
rgb = mat['image_rgb']
cv2.namedWindow("LOL")
while True:
    hsv = cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
    dark_blue = (90, 100, 20)
    light_blue = (110, 255, 255)
    dark_red = (0, 100, 50)
    light_red = (5, 255, 255)
    dark_green = (45, 100, 20)
    light_green = (75, 255, 255)
    maskb = cv2.inRange(hsv, dark_blue, light_blue)
    maskr = cv2.inRange(hsv,dark_red,light_red)
    maskg = cv2.inRange(hsv, dark_green, light_green)
    result = cv2.bitwise_and(rgb, rgb, mask=maskb)
    outb = cv2.connectedComponentsWithStats(maskb.astype(np.uint8))
    outr = cv2.connectedComponentsWithStats(maskr.astype(np.uint8))
    outg = cv2.connectedComponentsWithStats(maskg.astype(np.uint8))
    max = 0
    max_b = []
    for i in range(1, outb[0]):
        if max < outb[2][i][4]:
            max = outb[2][i][4]
            max_b = outb[2][i]
    max = 0
    max_r = []
    for i in range(1, outr[0]):
        if max < outr[2][i][4]:
            max = outr[2][i][4]
            max_r = outr[2][i]
    max = 0
    max_g = []
    for i in range(1, outg[0]):
        if max < outg[2][i][4]:
            max = outg[2][i][4]
            max_g = outg[2][i]
    print(max_g[0],max_g[1],max_g[0] + max_g[2],max_g[1]+max_g[3])

    im = rgb.copy()
    cv2.rectangle(img=im, pt1=(max_b[0], max_b[1]), pt2=(max_b[0] + max_b[2], max_b[1] + max_b[3]), color=(0, 255, 0), thickness=2)
    cv2.rectangle(img=im, pt1=(max_r[0], max_r[1]), pt2=(max_r[0] + max_r[2], max_r[1] + max_r[3]), color=(0, 0, 255),
                  thickness=2)
    cv2.rectangle(img=im, pt1=(max_g[0], max_g[1]), pt2=(max_g[0] + max_g[2], max_g[1] + max_g[3]), color=(255, 0, 255),
                  thickness=2)

    cv2.imshow("LOL",im)
    cv2.waitKey(1)