from functions import *


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


        detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        if event == "Face Detection":
            face_toggle = not face_toggle

        if event == "Grayscale":
            gray_toggle = not gray_toggle

        if event == "Sketch":
            sketch_toggle = not sketch_toggle

        if event == "Cartoonify":
            cartoon_toggle = not cartoon_toggle

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