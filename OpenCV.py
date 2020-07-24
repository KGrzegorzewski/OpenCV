import cv2
import imutils

#Loading an image
image = cv2.imread("test.png")
cv2.imshow("Image", image)

#HSV image
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv_image)
cv2.waitKey(0)

#HLS image
hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imshow("HLS", hls_image)
cv2.waitKey(0)


#BGR to GRAY
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray_image)

#Finding edges by Canny filter
edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=150, apertureSize=3)
cv2.imshow("Edged", edged_image)

#Thresholding
thresh = cv2.threshold(gray_image, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)

#Finding countours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

CNT_COLOUR = (240, 0, 159)

for counter in cnts:
    cv2.drawContours(output, [counter], -1, CNT_COLOUR, 3)
    text = "I found {} objects!".format(len(cnts))
    cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CNT_COLOUR, 2)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

#Erosions and dilations are typically used to reduce noise in binary images (a side effect of thresholding).

#Erosing - reducing the contour size with 5 iterations
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)

#Dilating - enlarging (...)
mask_dilating = thresh.copy()
mask_dilating = cv2.dilate(mask_dilating, None, iterations=5)
cv2.imshow("Dilated", mask_dilating)

#Masking
mask_image = thresh.copy()
masked_image = cv2.bitwise_and(image, image, mask=mask_image)
cv2.imshow("Output", masked_image)

cv2.waitKey(0)