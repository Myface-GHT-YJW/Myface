import cv2


def CatchPicture(path_name, catch_pic_num=500):
    window_name = "catch face"
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(0)
    cascade_path = "./haarcascades/haarcascade_frontalface_default.xml"
    num = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(cascade_path)
        # 识别出人脸数量
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(12, 12))

        if len(facerect) > 0:
            print('face detected')
            color = (255, 255, 255)  # 白
            for rect in facerect:
                x, y, w, h = rect
                img_name = '%s/%d.jpg' % (path_name, num)
                print(img_name)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)  # 照片写入到文件
                num += 1
                if num > (catch_pic_num):  # 如果超过指定最大保存数量退出循环
                    break

                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                # 显示当前捕捉到了多少人脸图片了
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (0, 255, 0), 2)
        # 超过指定最大保存数量结束程序
        if num > (catch_pic_num): break
        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(15)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    CatchPicture('./data/lwy', 50)