import os
import cv2
from PIL import Image, ImageDraw
from datetime import datetime
import time
import re


# detectFaces()返回图像中所有人脸的矩形坐标（矩形左上、右下顶点）
# 使用haar特征的级联分类器haarcascade_frontalface_default.xml，在haarcascades目录下还有其他的训练好的xml文件可供选择。
# 注：haarcascades目录下训练好的分类器必须以灰度图作为输入。
def DetectFaces(image_path):
    face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

    img = cv2.imread(image_path)
    # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
        #        gray,
        #        scaleFactor=1.2,
        #        minNeighbors=5,
        #        minSize=(30, 30),
    )

    result = []
    for (x, y, w, h) in faces:
        result.append((x, y, x + w, y + h))

    print("Found {0} faces!".format(len(faces)))
    return result


#    # Draw a rectangle around the faces
#    for (x, y, w, h) in faces:
#        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#    cv2.imshow("Faces found", image)
#    cv2.waitKey(0)

# 在原图像上画矩形，框出所有人脸。
# 调用Image模块的draw方法，Image.open获取图像句柄，ImageDraw.Draw获取该图像的draw实例，然后调用该draw实例的rectangle方法画矩形(矩形的坐标即
# detectFaces返回的坐标)，outline是矩形线条颜色(B,G,R)。
def DrawFaces(image_path):
    faces = DetectFaces(image_path)
    if faces:
        img = Image.open(image_path)
        draw_instance = ImageDraw.Draw(img)
        for (x1, y1, x2, y2) in faces:
            draw_instance.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))
        img.save(image_path.split(".")[0] + '_drawfaces' + ".jpg")


def DetectEyes(image_path):
    eye_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    faces = DetectFaces(image_path)

    img = cv2.imread(image_path)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    result = []
    #    eyes = eye_cascade.detectMultiScale(
    #        gray,
    #        scaleFactor=1.3,
    #        minNeighbors=2,
    #    )
    #    for (x,y,w,h) in eyes:
    #        result.append((x,y,x+w,y+h))
    for (x1, x2, y1, y2) in faces:
        roiGray = gray[y1:y2, x1:x2]
        eyes = eye_cascade.detectMultiScale(
            roiGray,
            scaleFactor=1.1,
            minNeighbors=3,
            #            cv2.CASCADE_SCALE_IMAGE,(2,2)
        )
        for (ex, ey, ew, eh) in eyes:
            result.append((x1 + ex, y1 + ey, x1 + ex + ew, y1 + ey + eh))

    print("Found {0} eyes!".format(len(eyes)))
    return result


def DrawEyes(image_path):
    eyes = DetectEyes(image_path)
    if eyes:
        img = Image.open(image_path)
        draw_instance = ImageDraw.Draw(img)
        for (x1, y1, x2, y2) in eyes:
            draw_instance.rectangle((x1, x2, y1, y2), outline=(0, 255, 0))
        img.save(image_path.split(".")[0] + '_draweyes' + ".jpg")


def DetectSmiles(image_path):
    smiles_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_smile.xml")
    faces = DetectFaces(image_path)

    img = cv2.imread(image_path)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    result = []
    smiles = smiles_cascade.detectMultiScale(gray, 4, 10)
    for (x, y, w, h) in smiles:
        result.append((x, y, x + w, y + h))
    #    for (x1,x2,y1,y2) in faces:
    #        roiGray = gray[y1:y2, x1:x2]
    #        smiles = smiles_cascade.detectMultiScale(
    #            roiGray,
    #            scaleFactor=1.1,
    #            minNeighbors=3,
    ##            cv2.CASCADE_SCALE_IMAGE,(2,2)
    #        )
    #        for (sx,sy,sw,sh) in smiles:
    #            result.append((x1+sx,y1+sy,x1+sx+sw,y1+sy+sh))

    print("Found {0} smiles!".format(len(smiles)))
    return result


def DrawSmiles(image_path):
    smiles = DetectSmiles(image_path)
    if smiles:
        img = Image.open(image_path)
        draw_instance = ImageDraw.Draw(img)
        for (x1, y1, x2, y2) in smiles:
            draw_instance.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))
        img.save(image_path.split(".")[0] + '_drawsmiles' + ".jpg")


def SaveFaces(image_path):
    faces = DetectFaces(image_path)
    if faces:
        # 将人脸保存在save_dir目录下。
        # Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        save_dir = image_path.split('.')[0] + "_faces"
        #        save_dir = re.split(r'/|\\|\\\|//', save_dir)[-1]
        if not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except OSError:
                pass

        count = 0
        for (x1, y1, x2, y2) in faces:
            fileName = os.path.join(save_dir, str(count) + ".jpg")
            Image.open(image_path).crop((x1, y1, x2, y2)).save(fileName)
            count += 1


if __name__ == "__main__":
    img = "D:/abba.png"
    #    img="75_26150317_1.jpg"
    #    img="5a72d6ee6b46f.jpg"
    # SaveFaces("abba.png")
    DrawFaces(img)
    DrawEyes(img)
    DrawSmiles(img)