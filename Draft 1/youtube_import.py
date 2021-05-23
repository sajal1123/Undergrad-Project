import PySimpleGUIQt as sg

sg.theme("DarkBlue17")

layout = [
    [
        sg.Text("Enter video URL:"),
        sg.InputText()
    ],
    [
        sg.Button("Detect Humans", key="-DETECT-")
    ],
]

window = sg.Window("URL Entry", layout)

while True:
    event, values = window.read()
    if event == "-DETECT-":
        print("URL: ", values[0])

        print("END")
        break

window.close()
print(values[0])