import cv2
from pic_train import Model
# from image_show import show_image
from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE


def FaceRecognize(file_path):
    cap = cv2.VideoCapture(0)
    # 人脸识别分类器本地存储路径
    cascade_path = "./haarcascades/haarcascade_frontalface_default.xml"
    # 加载模型
    model = Model()
    model.load_model(file_path)
    while True:
        ret, frame = cap.read()
        print(frame)

        ##图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(8, 8))

        if len(facerect) > 0:
            print('face detected')
            color = (255, 0, 0)  # 蓝
            for rect in facerect:
                # 截取脸部图像提交给模型识别这是谁
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]

                result = model.predict(image)
                if result == 0:  # boss
                    cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (255, 255, 255), 2)
                    print('Boss is approaching')
                    # cv2.imshow("识别朕", frame)
                #                    show_image()
                else:
                    print('Not boss')
        cv2.imshow("frame", frame)
        #        #10msec的带灯时间
        #        k = cv2.waitKey(10)
        #        #Esc退出
        #        if k == 27:
        #            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    name = 'chy'
    file_path = './store/model_' + name + '.h5'
    FaceRecognize(file_path)