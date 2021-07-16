import cv2.dnn
import os
#import emotion
from functions import *

# define the list of age buckets our age detector will predict
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-22)", "(25-34)",
               "(35-43)", "(44-59)", "(60+)"]
gender_list = ["Male", "Female"]

detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
print(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

prototxtPath = cv2.data.haarcascades+'deploy.prototxt.txt'
weightsPath = cv2.data.haarcascades+"res10_300x300_ssd_iter_140000.caffemodel"
print(prototxtPath)
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# load our serialized age detector model from disk
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([os.path.dirname(os.path.realpath(__file__)), "age_deploy.prototxt"])
weightsPath = os.path.sep.join([os.path.dirname(os.path.realpath(__file__)), "age_net.caffemodel"])
ageNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

prototxtPath = cv2.data.haarcascades+"deploy_gender.prototxt"
weightsPath = cv2.data.haarcascades+"gender_net.caffemodel"
gender_net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)


def main():
    sg.theme("DarkBlue17")

    feed_area = [

        [sg.Text("OpenCV", size=(60,1), justification="centre")],
        [sg.Image(filename="", key="-IMAGE-")],
    ]

    sliders = [

        [sg.Radio("None","Radio",True, size=(10,1))],
        [
            sg.Radio("brightness", "Radio", size=(15,1), key="-BRIGHT-"),
            sg.Slider((0, 255*2),255,1,orientation="h",size=(20,15),key="-BRIGHT VAL-")

        ],
        [
            sg.Radio("Saturation", "Radio", size=(15, 1), key="-BRIGHT NEW-"),
            sg.Slider((0, 30), 10, 1, orientation="h", size=(20, 15), key="-BRIGHT NEW VAL-")

        ],
        [
            sg.Radio("contrast", "Radio", size=(15,1), key="-CONT-"),
            sg.Slider((0, 127*2),127,1,orientation="h",size=(20,15),key="-CONT VAL-")

        ],

        [
            sg.Radio("threshold","Radio",size=(10,1), key="-THRESH-"),
            sg.Slider((0,255),128,1,orientation="h",size=(40,15),key="-THRESH SLIDER-")
        ],
        [
            sg.Radio("canny","Radio",size=(10,1),key="-CANNY-"),
            sg.Slider((0,255),128,1,orientation="h",size=(20,15),key="-CANNY SLIDER A-",),
            sg.Slider((0,255),128,1,orientation="h",size=(20,15),key="-CANNY SLIDER B-")
        ],
        [
            sg.Radio("blur","Radio",size=(10,1),key="-BLUR-"),
            sg.Slider((1,11),1,1,orientation="h",size=(40,15),key="-BLUR SLIDER-")
        ],
        [
            sg.Radio("hue","Radio",size=(10,1),key="-HUE-"),
            sg.Slider((0,255),0,1,orientation="h",size=(40,15),key="-HUE SLIDER-")
        ],
        [
            sg.Radio("enhance","Radio",size=(10,1),key="-ENHANCE-"),
            sg.Slider((1,255),128,1,orientation="h",size=(40,15),key="-ENHANCE SLIDER-")
        ],

        [sg.Text("Filters")],

        [
            sg.Button("Summer", size=(10,1))
        ],
        [
            sg.Button("Sepia", size=(10,1))
        ],
        [
            sg.Radio("Summer", "Radio", size=(10,1), key="-SUMMER-"),
            sg.Slider((0,100), 0, 1, orientation="h", size=(40,15), key="-SUMMER SLIDER-")
        ],
        [
            sg.Radio("Winter", "Radio", size=(10,1), key="-WINTER-"),
            sg.Slider((0,100), 0, 1, orientation="h", size=(40,15), key="-WINTER SLIDER-")
        ],
        [
            sg.Radio("Grain", "Radio", size=(10,1), key="-GRAIN-"),
            sg.Slider((0,100), 0, 1, orientation="h", size=(40,15), key="-GRAIN SLIDER-")
        ],
        [
            sg.Button("Pencil Sketch", size=(15,1)),
            sg.Radio("Color", "Radio", size=(10,1), key="-PENCIL COLOR-"),
            sg.Radio("Gray", "Radio", size=(10,1), key="-PENCIL GRAY-")
        ],

        [
            sg.Button('Age and Gender Detection', size=(25, 1)),
        ]
    ]

    buttons = [
        [sg.Button("Grayscale", size=(10,1))],
        [sg.Button("Extract Faces", size=(15,1))],
        [sg.Button("Sketch", size=(10,1))],
        [sg.Button("Cartoonify", size=(15,1))],
        [sg.Button("Save", size=(10,1))],
        [sg.Button("Face Detection", size=(20,1))],
        [sg.Button("Reset", size=(10,1))],
        [sg.Button("Exit",size=(10,1))]
    ]


    webcam_layout = [
        [sg.Column(feed_area, size=(500, 500)),
         sg.Column(sliders, size=(0, 100))],[
            sg.Column(buttons, size=(100, 100))]
    ]

    ''' 
    home_window = sg.Window("Home Page", home_layout)
    selection = ""
    while True:
        event, values = home_window.read(timeout=10)
        if event == "-VIDEO-":
            selection = "Webcam"
            print("Webcam selected - ", selection)
            break
        if event == "-IMAGE-":
            selection = "Image"
            print("Image selected")
            break

    if selection == "Webcam":
        cap = cv2.VideoCapture(0)
    elif selection == "Image":
        import image_viewer
        image_path = image_viewer.image
        #print(cap)
        cap = cv2.imread(image_path)
    #frame = cap
    '''
    ctr = 1
    window = sg.Window("OpenCV", layout=webcam_layout, location=(800,100))
    frame = cv2.VideoCapture(0)
    face_toggle = False
    gray_toggle = False
    sketch_toggle = False
    cartoon_toggle = False
    flag_sepia = False
    flag_summer = False
    flag_pencil = False
    flag_age = False
    flag_emotion = False
    flag_gender = False
    while True:
        event, values = window.read(timeout=10)
        if event == "Exit":
            window.close()
            import master
        if event == sg.WIN_CLOSED:
            break
        _, cap = frame.read()
        #if event == "Reset":
        #    cap = frame
        if values["-THRESH-"]:
            cap = cv2.cvtColor(cap, cv2.COLOR_BGR2LAB)
            cap = cv2.threshold(cap, values["-THRESH SLIDER-"],255,cv2.THRESH_BINARY)[1]
        elif values["-CANNY-"]:
            cap = cv2.Canny(cap, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"])
        elif values["-BLUR-"]:
            cap = cv2.GaussianBlur(cap,(21,21),values["-BLUR SLIDER-"])
        elif values["-HUE-"]:
            cap = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
            cap[:,:,0] += int(values["-HUE SLIDER-"])
            cap = cv2.cvtColor(cap, cv2.COLOR_HSV2BGR)
        elif values["-ENHANCE-"]:
            enh_val = values["-ENHANCE SLIDER-"]/40
            clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8,8))
            lab = cv2.cvtColor(cap, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            cap = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif values["-BRIGHT NEW-"]:
            cap = brightness(cap, (values["-BRIGHT NEW VAL-"])/10)
        elif values["-BRIGHT-"]:
            cap = controller(cap, values["-BRIGHT VAL-"], values["-CONT VAL-"])
        elif values["-CONT-"]:
            cap = controller(cap, values["-BRIGHT VAL-"], values["-CONT VAL-"])
        elif values["-SUMMER-"]:
            cap = Summer(cap, values["-SUMMER SLIDER-"]/100)

        elif values["-WINTER-"]:
            cap = winter(cap, values["-WINTER SLIDER-"]/100)

        elif values["-GRAIN-"]:
            cap = grain(cap, values["-GRAIN SLIDER-"]/100)


        if event == "Face Detection":
            face_toggle = not face_toggle

        if event == "Grayscale":
            gray_toggle = not gray_toggle

        if event == "Sketch":
            sketch_toggle = not sketch_toggle

        if event == "Cartoonify":
            cartoon_toggle = not cartoon_toggle

        if event == "Age and Gender Detection":
            flag_age = not(flag_age)
        #
        # if event == "Gender Detection":
        #     flag_gender = not(flag_gender)
        #
        # if event == "Emotion Detection":
        #     flag_emotion = not(flag_emotion)

        if event == "Summer":
            flag_summer = not(flag_summer)
        if flag_summer:
            #flag_summer = True
            cap = Summer(cap, 0.6)
        else:
            #cap = cv2.imread(image_path)
            cap = cap
            #flag_summer = False

        if event == "Sepia":
            flag_sepia = not(flag_sepia)
        if flag_sepia:
            cap = sepia(cap)
#                flag_sepia = True
        else:
            cap = cap
#                flag_sepia = False

        if event == "Pencil Sketch":
            flag_pencil = not(flag_pencil)

        if flag_pencil:
            if values["-PENCIL GRAY-"]:
                cap = pencil_gray(cap, False)
            elif values["-PENCIL COLOR-"]:
                cap = pencil_gray(cap, True)

        if event == "Extract Faces":
            gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale3(gray, scaleFactor=1.08, minNeighbors=5,
                                               minSize=(30, 30), outputRejectLevels=True)

            if len(faces) > 0:
                for (x,y,w,h) in faces[0]:
                    cv2.rectangle(cap, (x,y), (x+w,y+h), (0,200,0), 2)
                    temp_face = cap[y:y+h, x:x+w]
                    cv2.imshow("Face", temp_face)
        if face_toggle:
            gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale3(gray, scaleFactor=1.08, minNeighbors=5,
                                               minSize=(30,30), outputRejectLevels=True)

            if len(faces) > 0:
                rects = faces[0]
                for (x,y,w,h) in rects:
                    cv2.rectangle(cap, (x,y), (x+w,y+h), (0,200,0), 3)

        if gray_toggle:
            cap = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

        if sketch_toggle:
            cap = adjust_brightness(cap)
            gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            smooth = cv2.GaussianBlur(inv, (5, 5), sigmaX=0, sigmaY=0)

            cap = doge(inv, smooth)

        if cartoon_toggle:
            cap = cartoon(cap)

        if flag_age:
            image = cap
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            print("[INFO] computing face detections...")
            faceNet.setInput(blob)
            detections = faceNet.forward()
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > 0.2:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the ROI of the face and then construct a blob from
                    # *only* the face ROI
                    face = image[startY:endY, startX:endX]
                    try:
                        ageBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                         (78.4263377603, 87.7689143744, 114.895847746),
                                                         swapRB=False)

                        genderBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                        (78.4263377603, 87.7689143744, 114.895847746),
                                                        swapRB=False)
                    except:
                        pass

                # make predictions on the age and find the age bucket with
                    # the largest corresponding probability
                    try:
                        ageNet.setInput(ageBlob)
                        preds = ageNet.forward()
                        i = preds[0].argmax()
                        age = AGE_BUCKETS[i]
                        ageConfidence = preds[0][i]

                        #detecting gender
                        gender_net.setInput(genderBlob)
                        gender_preds = gender_net.forward()
                        gender = gender_list[gender_preds[0].argmax()]
                        print("Gender - ", gender)


                    # display the predicted age to our terminal
                        text = "{} {}: {:.2f}%".format(gender, age, ageConfidence * 100)
                        print("[INFO] {}".format(text))

                        # draw the bounding box of the face along with the associated
                        # predicted age
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(image, (startX, startY), (endX, endY),
                                      (0, 0, 255), 2)
                        cv2.putText(image, text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                    except:
                        pass

    #
#         if flag_gender:
#             img = cap[:,:,::-1]
#             face_locations = emotion.detect_face(img)
#             # Display the results
#             for top, right, bottom, left, sex_preds, age_preds, emotion_preds in face_locations:
#                 # Draw a box around the face
#                 cv2.rectangle(cap, (left, top), (right, bottom), (0, 0, 255), 2)
#
#                 sex_text = 'Female' if sex_preds > 0.5 else 'Male'
#                 cv2.putText(cap, 'Gender: {:.3f}'.format(sex_preds), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
#
#         if flag_emotion:
#             img = cap[:,:,::-1]
# #            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
#             face_locations = emotion.detect_face(img)
#             # Display the results
#             for top, right, bottom, left, sex_preds, age_preds, emotion_preds in face_locations:
#                 # Draw a box around the face
#                 cv2.rectangle(cap, (left, top), (right, bottom), (0, 0, 255), 2)
#
#                 cv2.putText(cap, 'Emotion: {:.3f}'.format(emotion_preds), (left, top-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)


        '''
        try:
            height, width = cap.shape[:2]
            max_height, max_width = 700, 700
            if max_height < height or max_width < width:
                scaling_factor = max_height / float(height)
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width/float(width)
                cap = cv2.resize(cap, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        except:
            pass
        '''
        if event == "Save":
            cv2.imwrite("Enhanced {}.jpg".format(ctr), cap)
            ctr += 1

        imgbytes = cv2.imencode(".jpg", cap)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()
main()