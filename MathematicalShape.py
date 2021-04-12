import cv2
import numpy as np

'''Hough Transform
Used to detect any shape, if we can define it mathematically
It can detect the shape if it is broken or distorted little bit
Example
A line in Cartesian Coordinate System is defined as y = mx + c
In polar coordinate system xcos(theta) + ysin(theta) = r
In Hough transform line y = mx + c is represented in m-c space instead of x-y space, so line can be
represented as a point
Becoz in Coordinate Line is a collection of point, while in Hough Space it is a single point
So management is easy

Similarly, a point in xy space lets say x1,y1 can be represented as line in Hough Space
c = -x1m + y1 with slope = -x1 and intercept y1

So a line passing through points in Cartesian Product is represented as lines intersecting a point in
MC Space

Similarly, in polar coordinate r = x cos(theta)  + y sin(theta) or
y = - cos(theta)/sin(theta) + r/ sin(theta)  x-y space is transformed to theta - r space as a point

and a point in xy space is represented as sin wave in hough transform

Usually, Cartesian to Hough is difficult for Vertical lines so we will use Polar coordinates

Steps:
1. Edge Detection like using Canny Edge detector
2. Mapping of edge points to Hough Space and storage in accumulator
3. Interpretation of the accumulator to yield lines of infinite length using some thresholds or other
4. Conversion of infinite lines into finite lines

OpenCV uses two types of Hough Transforms
1. Standard    2 Probabilistic '''

def stdHoughT():
    img = cv2.imread('sudoku.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize= 3)
    cv2.imshow("Edges", edges)  # just checking the edges
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    ''' params
    1. source image
    lines is output vector of lines. Each line is represented by a 2 or 3 element vector (rho, theta) or
    (rho, theta, votes). rho is the distance from the coordinate (0,0) to top left coordinate of image
    theta: line rotation angle in radians, votes is the value of accumulator
    2. rho: Distance resolution of accumulator in pixel
    3. theta: Angel resolution of accumulator in radians
    4. Threshold: Accumulator param, only those lines will be returned that get enough votes ( > threshold)'''

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho   #(x0,y0) is origin
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def probHoughT():
    ''' previous functions, we are getting lines of infinite lengths
    Let's solve it by using Probabilistic transform'''
    img = cv2.imread('sudoku.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow("Edges", edges)  # just checking the edges
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    #we are not taking all the points here
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()