import PySimpleGUI as sg

#sg.Window(title="Hello World", layout=[[]], margins=(300,300)).read()

layout = [[sg.Text("Hello BOIIIIIIII")],[sg.Button("BOII")]]

window = sg.Window("Demo", layout, margins=(300,300))

while True:
    event, values = window.read()
    if event == "BOII" or event == sg.WIN_CLOSED:
        break
    window.close()
