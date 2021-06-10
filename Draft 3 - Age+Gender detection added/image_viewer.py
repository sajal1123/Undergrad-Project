import PySimpleGUIQt as sg
import os.path
import cv2

sg.theme("DarkBlue17")

file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse()
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(20, 20), key="-FILE LIST-"
        )
    ]
]

image_viewer_column = [
    [sg.Text("Choose an image from list:")],
    [sg.Text(size=(20,1), key="-TOUT-")],
    [sg.Button("Select Image", size=(12, 1), key="-SAVE-")],
    [sg.Image(key="-IMAGE-")]
]

layout = [
    [
        sg.Column(file_list_column, size=(100,100)),
        sg.VerticalSeparator(),
        sg.Column(image_viewer_column,size=(100,300))
    ]
]

window = sg.Window("Image Viewer", layout)

while True:
    event, values = window.read()
    if event == "Exit":
        window.close()
        import master
    if event == sg.WIN_CLOSED:
        break

    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            window["-TOUT-"].update(filename)
            img = cv2.imread(filename)
            try:
                height, width = img.shape[:2]
                print("Height:", height, "\nWidth:", width)
                max_height, max_width = 700, 700
                if max_height < height or max_width < width:
                    scaling_factor = max_height / float(height)
                    if max_width/float(width) < scaling_factor:
                        scaling_factor = max_width/float(width)
                    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            except:
                pass
            imgbytes = cv2.imencode(".jpg", img)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)

#            window["-IMAGE-"] = img
        except:
            pass
    if event == "-SAVE-":
        image = filename
        break
window.close()


