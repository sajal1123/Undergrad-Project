
import PySimpleGUIQt as sg


def main():
    sg.theme("DarkBlue17")
    home_layout = [
        [sg.Text("Welcome!")],
        [sg.Button("Live Feed", size=(20, 1), key="-VIDEO-")],
        [sg.Button("Input Image", size=(20, 1), key="-IMAGE-")],
        [sg.Button("Color Detection", size=(20,1), key="-COLOR-")],
        [sg.Button("Human Detection(Image)", size=(20,1), key="-HUMAN IMAGE-")],
        [sg.Button("Human Detection(Video)", size=(20,1), key="-HUMAN VIDEO-")],
        [sg.Button("Human Detection(Online)", size=(20,1), key="-HUMAN VIDEO YT-")]
    ]
    home_window = sg.Window("Home Page", home_layout)
    selection = ""
    while True:
        event, values = home_window.read(timeout=10)
        if event == "-VIDEO-":
            selection = "Webcam"
            print("Webcam selected - ", selection)
            import video
            break
        if event == "-IMAGE-":
            selection = "Image"
            print("Image selected")
            import image
        if event == "-COLOR-":
            selection = "Color Detection"
            print(selection)
            import color_detection
        if event == "-HUMAN IMAGE-":
            selection = "Human Detection Image"
            print(selection)
            import human_detector_image
            break
        if event == "-HUMAN VIDEO-":
            selection = "Human Detection Video"
            print(selection)
            import human_detection_video
        if event == "-HUMAN VIDEO YT-":
            selection = "Human Detection Online"
            print(selection)
            import human_detection_youtube
main()
