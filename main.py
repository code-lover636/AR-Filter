import cv2
import cvzone

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('hfd.xml')
item = cv2.imread('crown.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        if y - 150 >= 0 and x + w <= frame.shape[1]:
            crop_img = frame[y:y+h, x:x+w]
            
            imgRGB = cv2.resize(item, (w, h))
            
            frame = cvzone.overlayPNG(frame, imgRGB, [x, y-150])

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
