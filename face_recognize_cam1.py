 import face_recognition
import cv2

cap = cv2.VideoCapture(0)

img_name=["sgl.jpg", "chy.jpg"]
symbol_img=[]
symbol_face_encoding=[]
for i, x in enumerate(img_name, 0):
    symbol_img.append(face_recognition.load_image_file(x))
    symbol_face_encoding.append(face_recognition.face_encodings(symbol_img[i])[0])

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = cap.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if process_this_frame:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        match=[]
        for face_encoding in face_encodings:
            for i, x in enumerate(symbol_face_encoding, 0):
                match = face_recognition.compare_faces([x], face_encoding, 0.8)

                if match[0]:
                    name = img_name[i].split('.')[0]
                else:
                    name = "unknown"
    
                face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),  2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()