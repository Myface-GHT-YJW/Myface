# 识别图片中的人脸
import face_recognition


def FaceRecognitionPic(img1, img2, img_unknow):
    image_1 = face_recognition.load_image_file(img1);
    image_2 = face_recognition.load_image_file(img2);
    unknown_image = face_recognition.load_image_file(img_unknow);

    encoding_1 = face_recognition.face_encodings(image_1)[0]
    encoding_2 = face_recognition.face_encodings(image_2)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([encoding_1, encoding_2], unknown_encoding)
    labels = [img1.split('.')[0], img2.split('.')[0]]
    print('results:' + str(results))

    for i in range(0, len(results)):
        if results[i] == True:
            print('The person is: ' + labels[i])
            return labels[i]


if __name__ == "__main__":
    img1_path = "jobs.jpg"
    img2_path = "obama.jpg"
    img_unknow_path = "obama1.jpg"
    FaceRecognitionPic(img1_path, img2_path, img_unknow_path)