import cv2, imutils
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
#img = cv2.imread('C:/Users/Asus/Downloads/receipt-scanned.jpg')
path = 'G:/SDHC/Camera/IMG_20161001_142124.jpg'
def scan(path):
        image = cv2.imread(path)
        #image = imutils.resize(image, 500)

        gray = cv2.cvtColor(image, 6)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 100, 200)

        #cv2.imshow('img', edged)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts=imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        screenCnt = None
        for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                print(len(approx))
                if len(approx)==4:
                        screenCnt = approx
                        break


        if screenCnt is not None:
                paper = four_point_transform(image, screenCnt.reshape((4, 2)))
                warped = four_point_transform(gray, screenCnt.reshape((4, 2)))
        else:
                warped = gray
        T = threshold_local(warped, 11, offset=10)

        warped = (warped > T).astype("uint8") * 255

        #cv2.imshow('img', imutils.resize(warped, 600))
        return warped
        
