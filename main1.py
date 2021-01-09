
import face_detect_pic
import face_detect_cam
import face_recognize_pic
import face_recognize_cam
import catch_picture
import pic_train
import face_recognize_cam_via_model
import cv2

#
if __name__ == "__main__":
    print("1.照片人脸检测")
    print("2.视频人脸检测")
    print("3.照片人脸识别")
    print("4.视频人脸识别")
    print("5.采集训练数据")
    print("6.训练模型")
    print("7.评估模型")
    print("8.视频人脸识别（通过模型）")

    op = int(input("输入选项："))

    if op == 1:
        img = input("输入图片路径：")
        face_detect_pic.DrawFaces(img)
        face_detect_pic.DrawEyes(img)
        face_detect_pic.DrawSmiles(img)
        print("completed")
    elif op == 2:
        face_detect_cam.FaceDetect()
    elif op == 3:
        img1 = input("输入样本1路径：")
        img2 = input("输入样本2路径：")
        img_unknow = input("输入检测样本路径：")
        face_recognize_pic(img1, img2, img_unknow)
    elif op == 4:
        img = input("输入目标图片路径：")
        tolerance = float(input("输入检测精度："))
        face_recognize_cam.FaceRecognize(img, tolerance)
    elif op == 5:
        num = int(input("采集图片数量："))
        save_dir = input("保存路径：")
        catch_picture.CatchPicture(save_dir, num)
    elif op == 6:
        name = input("输入训练文件夹名称：")
        dataset = pic_train.Dataset("./data/", name)
        dataset.load()
        model = pic_train.Model()
        model.build_model(dataset)
        model.train(dataset)
        model.save_model(file_path='./store/model_' + name + '.h5')

    elif op == 7:
        name = input("输入训练文件夹名称：")
        dataset = pic_train.Dataset("./data/", name)
        dataset.load()
        model = pic_train.Model()
        model.load_model(file_path='./store/model_' + name + '.h5')
        model.evaluate(dataset)
    elif op == 8:
        name = input("输入训练文件夹名称：")
        face_recognize_cam_via_model.FaceRecognize(file_path='./store/model_' + name + '.h5')