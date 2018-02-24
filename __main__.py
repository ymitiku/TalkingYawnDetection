import cv2
import os
import dlib

detector = dlib.get_frontal_face_detector()

face_cascade = cv2.CascadeClassifier("haarcas/haarcascade_profileface.xml")

current_dir = "/dataset/yawn/splited-100/5-FemaleGlasses-Talking-0"
for img_file in os.listdir(current_dir):
    img = cv2.imread(os.path.join(current_dir,img_file))
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
    faces = face_cascade.detectMultiScale(img_gray,1.5,5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces)==0:
        faces =detector(img_gray)
        if len(faces)==0:
            continue
        face = faces[0]
        cv2.rectangle(img,(face.left(),face.top()),(face.right(),face.bottom()),(0,0,255),2)
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
