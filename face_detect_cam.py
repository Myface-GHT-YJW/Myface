import cv2
import numpy as np
from PIL import Image

#time.sleep(3)

def FaceDetect():
    faceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    smilesCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_smile.xml")
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        print(frame)
    #    cv2.imshow("capture", frame)
        
        #Our operations on the frame come here
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
    #    # 识别输入图片中的人脸对象.返回对象的矩形尺寸
    #    # 函数原型detectMultiScale(gray, 1.2,3,CV_HAAR_SCALE_IMAGE,Size(30, 30))
    #    # gray需要识别的图片
    #    # 1.1：表示每次图像尺寸减小的比例
    #    # 5：表示每一个目标至少要被检测到4次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)
    #    # CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的最小最大尺寸
    #    # faces：表示检测到的人脸目标序列
        faces = faceCascade.detectMultiScale(
           gray,
           scaleFactor=1.2,
           minNeighbors=5,
    #       minSize=(30, 30),
        )
        if len(faces)>0:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
                roi_gray = gray[y:(y+h),x:(x+w)]
                roi_color = frame[y:(y+h),x:(x+w)]
    
                eyes = eyeCascade.detectMultiScale(roi_gray,1.4,3,cv2.CASCADE_SCALE_IMAGE,(2,2))
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                smiles = smilesCascade.detectMultiScale(roi_gray,5,25,cv2.CASCADE_SCALE_IMAGE,(2,2))
                for (sx,sy,sw,sh) in smiles:
                    cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    ##Display the resulting frame
    ##cv2.imshow('frame',gray)
    ##when everything done ,release the capture
    cap.release()
    cv2.destroyAllWindows() 
    ##cv2.destoryAllWindows()

if __name__ == "__main__":
    FaceDetect()