import face_detect_pic
import face_detect_cam
import face_recognize_pic
import face_recognize_cam
import catch_picture
import pic_train
import face_recognize_cam_via_model
import tkinter as tk

window = tk.Tk()  # 建立一个窗口
window.title('人脸')
# window.geometry('350x150')
#

bon = False


def op_1():
    top = tk.Tk()
    top.geometry('300x200')
    L1 = tk.Label(top, text="输入图片路径")
    L1.pack()
    E2 = tk.Entry(top, width=50)
    E2.pack()

    def show():
        print(E2.get())
        face_detect_pic.DrawFaces(E2.get())
        face_detect_pic.DrawEyes(E2.get())
        face_detect_pic.DrawSmiles(E2.get())
        a = tk.Tk()
        l2 = tk.Label(a, text='ok')
        l2.pack()
        a.mainloop()

    b2 = tk.Button(top, text='ok', command=show)
    b2.pack()
    top.mainloop()


def op_2():
    #    top=tk.Tk()
    #    top.geometry('300x200')
    face_detect_cam.FaceDetect()


#    top.mainloop()
def op_3():
    top = tk.Tk()
    top.geometry('600x400')
    L1 = tk.Label(top, text="输入样本1路径")
    L1.pack()
    E1 = tk.Entry(top, width=50)
    E1.pack()
    L2 = tk.Label(top, text="输入样本2路径")
    L2.pack()
    E2 = tk.Entry(top, width=50)
    E2.pack()
    L3 = tk.Label(top, text="输入检测样本路径：")
    L3.pack()
    E3 = tk.Entry(top, width=50)
    E3.pack()

    def show():
        answer = face_recognize_pic.FaceRecognitionPic(E1.get(), E2.get(), E3.get())
        a = tk.Tk()
        a.geometry('300x200')
        l2 = tk.Label(a, text='the person is ' + answer)
        l2.pack()
        a.mainloop()

    b2 = tk.Button(top, text='ok', command=show)
    b2.pack()
    top.mainloop()


def op_4():
    top = tk.Tk()
    top.geometry('300x200')
    L1 = tk.Label(top, text="输入目标样本路径")
    L1.pack()
    E1 = tk.Entry(top, width=50)
    E1.pack()
    L2 = tk.Label(top, text="输入检测容忍度")
    L2.pack()
    E2 = tk.Entry(top, width=50)
    E2.pack()

    def show():
        face_recognize_cam.FaceRecognize(E1.get(), float(E2.get()))

    b1 = tk.Button(top, text='ok', command=show)
    b1.pack()
    top.mainloop()


def op_5():
    top = tk.Tk()
    top.geometry('300x200')
    L1 = tk.Label(top, text="采集图片数量")
    L1.pack()
    E1 = tk.Entry(top, width=50)
    E1.pack()
    L2 = tk.Label(top, text="目标名字")
    L2.pack()
    E2 = tk.Entry(top, width=50)
    E2.pack()

    def show():
        answer = './data/' + E2.get()
        catch_picture.CatchPicture(answer, int(E1.get()))

    b1 = tk.Button(top, text='ok', command=show)
    b1.pack()
    top.mainloop()


def op_6():
    top = tk.Tk()
    top.geometry('300x200')
    L1 = tk.Label(top, text="输入目标名字")
    L1.pack()
    E1 = tk.Entry(top, width=50)
    E1.pack()

    def show():
        dataset = pic_train.Dataset("./data/", E1.get())
        dataset.load()
        model = pic_train.Model()
        model.build_model(dataset)
        model.train(dataset)
        model.save_model(file_path='./store/model_' + E1.get() + '.h5')
        a = tk.Tk()
        l2 = tk.Label(a, text='ok')
        l2.pack()
        a.mainloop()

    b1 = tk.Button(top, text='ok', command=show)
    b1.pack()
    top.mainloop()


def op_7():
    top = tk.Tk()
    top.geometry('300x200')
    L1 = tk.Label(top, text="输入目标名字")
    L1.pack()
    E1 = tk.Entry(top, width=50)
    E1.pack()

    def show():
        dataset = pic_train.Dataset("./data/", E1.get())
        dataset.load()
        model = pic_train.Model()
        model.load_model(file_path='./store/model_' + E1.get() + '.h5')
        model.evaluate(dataset)
        a = tk.Tk()
        l2 = tk.Label(a, text='ok')
        l2.pack()
        a.mainloop()

    b1 = tk.Button(top, text='ok', command=show)
    b1.pack()
    top.mainloop()


def op_8():
    top = tk.Tk()
    top.geometry('300x200')
    L1 = tk.Label(top, text="输入目标名字")
    L1.pack()
    E1 = tk.Entry(top, width=50)
    E1.pack()

    def show():
        face_recognize_cam_via_model.FaceRecognize(file_path='./store/model_' + E1.get() + '.h5')

    b1 = tk.Button(top, text='ok', command=show)
    b1.pack()
    top.mainloop()


## 设置按钮
b1 = tk.Button(text='照片人脸检测', width=20, height=1, command=op_1).grid(row=2, sticky=tk.W)

b2 = tk.Button(text='视频人脸检测', width=20, height=1, command=op_2).grid(row=2, column=1, sticky=tk.E)

b3 = tk.Button(text='照片人脸识别', width=20, height=1, command=op_3).grid(row=4, sticky=tk.W)

b4 = tk.Button(text='视频人脸识别', width=20, height=1, command=op_4).grid(row=4, column=1, sticky=tk.E)

b5 = tk.Button(text='采集训练数据', width=20, height=1, command=op_5).grid(row=6, sticky=tk.W)

b6 = tk.Button(text='训练模型', width=20, height=1, command=op_6).grid(row=6, column=1, sticky=tk.E)

b7 = tk.Button(text='评估模型', width=20, height=1, command=op_7).grid(row=8, sticky=tk.W)

b8 = tk.Button(text='视频人脸识别（通过模型）', width=20, height=1, command=op_8).grid(row=8, column=1, sticky=tk.E)

window.mainloop()  # 循环，时刻刷新窗口

# if __name__ == "__main__":
#    print("1.照片人脸检测")
#    print("2.视频人脸检测")
#    print("3.照片人脸识别")
#    print("4.视频人脸识别")
#    print("5.采集训练数据")
#    print("6.训练模型")
#    print("7.评估模型")
#    print("8.视频人脸识别（通过模型）")
#
#    op = int(input("输入选项："))
#
#    if op == 1:
#        img = input("输入图片路径：")
#        face_detect_pic.DrawFaces(img)
#        face_detect_pic.DrawEyes(img)
#        face_detect_pic.DrawSmiles(img)
#        print("completed")
#    elif op == 2:
#        face_detect_cam.FaceDetect()
#    elif op == 3:
#        img1 = input("输入样本1路径：")
#        img2 = input("输入样本2路径：")
#        img_unknow = input("输入检测样本路径：")
#        face_recognize_pic(img1,img2,img_unknow)
#    elif op == 4:
#        img = input("输入目标图片路径：")
#        tolerance = input("输入检测精度：")
#        face_recognize_cam.FaceRecognize(img, tolerance)
#    elif op == 5:
#        num = int(input("采集图片数量："))
#        save_dir = input("保存路径：")
#        catch_picture.CatchPicture(save_dir, num)
#   elif op == 6:
#        name = input("输入训练文件夹名称：")
#        dataset = pic_train.Dataset("./data/", name)
#        dataset.load()
#        model = pic_train.Model()
#        model.build_model(dataset)
#        model.train(dataset)
#        model.save_model(file_path = './store/model_'+name+'.h5')
#
#    elif op == 7:
#        name = input("输入训练文件夹名称：")
#        dataset = pic_train.Dataset("./data/", name)
#        dataset.load()
#        model = pic_train.Model()
#        model.load_model(file_path = './store/model_'+name+'.h5')
#        model.evaluate(dataset)
#    elif op == 8:
#        name = input("输入训练文件夹名称：")
#        face_recognize_cam_via_model.FaceRecognize(file_path = './store/model_'+name+'.h5')