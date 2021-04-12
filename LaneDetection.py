import cv2
import numpy as np
from matplotlib import pyplot as plt

def region_of_interst(img, vertices):
    mask = np.zeros_like(img)
    ''' becoz we read gray image  and detected edges before, so no need to detect the channel'''
    #channel_count = img.shape[2]
    #match_mast_color = (255,) * channel_count   # creating the match color of the mask as original image

    match_mast_color = 255
    print(match_mast_color)
    cv2.fillPoly(mask, vertices, match_mast_color)
    mask_image = cv2.bitwise_and(img, mask)
    return  mask_image

def drawlines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def detectLane():
    image = cv2.imread('road3.jpg')
    # converting to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    #define region of interest (the lane i like to go)
    ''' ROI is a reactagle right from bottom of the image to  some where in the middle where to lane
    seems to merge at horizon'''
    region_of_interest_vertices = [
        (0, height), (width/2, height/2), (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interst(canny_image, np.array([region_of_interest_vertices], np.int32))

    #cropped_image = region_of_interst(image, np.array([region_of_interest_vertices], np.int32))
    ''' Since this line of code will detect edges of masked region of interest too. So we will put this code before cropping'''
    #gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
    #canny_image= cv2.Canny(gray_image, 100, 200)

    #now Houghline transform
    lines = cv2.HoughLinesP(cropped_image, rho=6, theta= np.pi/60, threshold= 160, lines= np.array([]), minLineLength=40, maxLineGap=25)

    image_with_lines = drawlines(image, lines)
    plt.imshow(image_with_lines)
    plt.show()


def processVideo(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    # define region of interest (the lane i like to go)
    region_of_interest_vertices = [
        (0, height), (width / 2, height / 2), (width, height)
    ]


    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 150)
    cropped_image = region_of_interst(canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 60, threshold=50, lines=np.array([]), minLineLength=40,
                        maxLineGap=25)

    image_with_lines = drawlines(image, lines)
    return image_with_lines


def detectLaneVideo():
    cap = cv2.VideoCapture('test_video.mp4')

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = processVideo(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


