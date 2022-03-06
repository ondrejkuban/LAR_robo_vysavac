import cv2
import scipy.io
import numpy as np

# tresholds
dark_blue = (90, 100, 20)
light_blue = (110, 255, 255)
dark_red = (0, 100, 50)
light_red = (5, 255, 255)
dark_green = (45, 70, 20)
light_green = (75, 255, 255)
#init
mat = scipy.io.loadmat('2022-03-03-15-32-23.mat')
rgb = mat['image_rgb']
depth = mat['image_depth']
cv2.namedWindow("RGB")
cv2.namedWindow("DEPTH")
while True:

    #convert to hsv
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    #creating mask to find rectangles
    maskb = cv2.inRange(hsv, dark_blue, light_blue)
    maskr = cv2.inRange(hsv, dark_red, light_red)
    maskg = cv2.inRange(hsv, dark_green, light_green)
    result = cv2.bitwise_and(rgb, rgb, mask=maskb)

    #finding rectangles
    outb = cv2.connectedComponentsWithStats(maskb.astype(np.uint8))
    outr = cv2.connectedComponentsWithStats(maskr.astype(np.uint8))
    outg = cv2.connectedComponentsWithStats(maskg.astype(np.uint8))

    #finding biggest rectangles for each color
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
    print(max_g[0], max_g[1], max_g[0] + max_g[2], max_g[1] + max_g[3])

    #drawing rectangle
    im = rgb.copy()
    cv2.rectangle(im, (max_b[0], max_b[1]), (max_b[0] + max_b[2], max_b[1] + max_b[3]), (0, 255, 0), 2)
    cv2.rectangle(im, (max_r[0], max_r[1]), (max_r[0] + max_r[2], max_r[1] + max_r[3]), (0, 0, 255),
                  2)
    cv2.rectangle(im, (max_g[0], max_g[1]), (max_g[0] + max_g[2], max_g[1] + max_g[3]), (255, 0, 255),
                  2)
    #finding distance to green
    dep = []
    for i in range(max_g[0], max_g[2] + max_g[0]):
        for j in range(max_g[1], max_g[3] + max_g[1]):
            dep.append(depth[i][j])
    cv2.putText(im, str(np.median(dep)), (max_g[0], max_g[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.imshow("RGB", im)
    cv2.imshow("DEPTH", depth)
    cv2.waitKey(1)
