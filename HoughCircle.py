import cv2
import numpy as np

def detectCircle():
    img = cv2.imread('smarties.png')
    output = img.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    #hough circle works good with blur images
    gray = cv2.medianBlur(gray, 5)
    circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1= 50, param2= 30, minRadius=0, maxRadius=0)
    ''' O/P: output vector of found circles
    Params are
    1: image: 8 bit dtype, single-channel, grayscale
    2: method: Detection Method: More info HoughModes, Currently HOUGH_GRADIENT is only implemented
    3: dp: inverse ratio of the accumulator resolution to the image resolution
    4: minDist: min dist between centers of detected circle
    5: param1: Method specific, For Hough_gradient it is higher threshold for canny edge detector
    6: param2: Method specific, For Hough_gradient, its is the accumulator threshold for the circle center at detectin stage
    7: minRadius
    8: maxRadius: if <=0 uses the maximum image dimension. if < 0, returns center without finding the radius '''

    # now find the coordinates of center of the circle and radius to draw
    detected_circle = np.uint16(np.around(circle))
    for (x, y, r) in detected_circle[0, :]:
        cv2.circle(output, (x, y), r, (0, 255, 0), 3)

    cv2.imshow('Output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()