import cv2
import numpy as np

''' Template matching is like finding a template object (small Object) in an image'''

def applytemplate():
    img = cv2.imread('messi5.jpg')
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('messi_face.JPG', 0)
    w, h = template.shape[::-1]  # becoz it returns h, w to reversing -1
    res = cv2.matchTemplate(grey_image, template, cv2.TM_CCOEFF_NORMED)
    print(res)  # the brightest point will be the top left corner of the image

    #so let the threshold will be 0.9
    threshold = 0.9
    loc = np.where( res >= threshold)
    print(loc)

    #draw rectangle # more the points, more the rectangles and we get thicker rectangle
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] +w, pt[1] + h), (0,0,255), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()