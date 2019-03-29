import cv2 as cv

original_image = cv.imread('figs/img3.jpg')
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')

detected_faces = face_cascade.detectMultiScale(
        grayscale_image,
        scaleFactor=100,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )

cv.imshow('Image', original_image)
cv.waitKey(0)
cv.destroyAllWindows()
