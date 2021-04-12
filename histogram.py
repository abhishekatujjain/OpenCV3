import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawHist():
    ''' histogram will help to understand the image: intensity, contrast'''
    ''' This function creates a black image from numpy and 
    plots histogram on the number of pixels with color values
     It will create only one hist becoz all pixel 200x200 are black'''
    img = np.zeros((200,200), np.uint8)
    cv2.imshow("Image", img)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawHist1():

    img = np.zeros((200,200), np.uint8)
    cv2.rectangle(img, (0, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(img, (0, 50), (100, 100), (127), -1)
    cv2.imshow("Image", img)
    plt.hist(img.ravel(), 256, [0, 256])


    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawHist2():
    img = cv2.imread('lena.jpg', 0)
    cv2.imshow("Image", img)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawHist3():
    img = cv2.imread('lena.jpg')
    b, g, r = cv2.split(img)
    cv2.imshow("Image", img)
    cv2.imshow("Blue", b)
    cv2.imshow("Green", g)
    cv2.imshow("Red", r)
    plt.hist(b.ravel(), 256, [0, 256])
    plt.hist(g.ravel(), 256, [0, 256])
    plt.hist(r.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawHistusingCV():
    img = cv2.imread('lena.jpg', 0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    ''' 2nd param: Channel, 0 becoz we are reading grayscale here, otherwise 3 channels are there
    3rd: We are applying no mask, 4th Size, 5th range
    '''
    plt.plot(hist)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()