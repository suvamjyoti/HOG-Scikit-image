import cv2
# pip install scikit-image for the hog function
from skimage.feature import hog

image = cv2.imread("Go1.jpg")
# original image size
C, W, H = image.shape[::-1]
# resizing factor
r = 5
# resizing the image
newimg = cv2.resize(image, (int(W/r), int(H/r)))

# main hog function
fd, hog_image = hog(newimg, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(8, 8), visualize=True, multichannel = True)

# finding the height and width of hog_image
h, w = hog_image.shape[::-1]

# manipulating the pixel data for better image storage
for a in range(w):
    for b in range(h):
        hog_image[a][b] = 10*hog_image[a][b]

# writing file to disk
cv2.imwrite("hog.jpg", hog_image)
# retrieving file from disk for better result
img = cv2.imread("hog.jpg")
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()