import PySimpleGUIQt as sg
import cv2
import numpy as np


def controller(img, brightness=255,
               contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    # putText renders the specified text string in the image.
    '''cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    '''
    return cal


def adjust_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 0
    v[(v - 10) < lim] = 0
    v[(v - 20) > lim] -= 20
    final = cv2.merge((h, s, v))
    img = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
    return img


def doge(x, y):
    return cv2.divide(x, 255 - y, scale=256)


def cartoon(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.medianBlur(gray, 5)
    edge = cv2.adaptiveThreshold(smooth, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon_img = cv2.bitwise_and(color, color, mask=edge)
    return cartoon_img

#New Functions


def brightness(img, val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*val
    hsv[:,:,1][hsv[:,:,1]>255] = 255
    hsv[:,:,2] = hsv[:,:,2]*val
    hsv[:,:,2][hsv[:,:,2]>255] = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def gamma(channel, gam):
    try:
        invGamma = 1/gam
    except:
        invGamma = 1/(gam+0.1)
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0, 256)]).astype("uint8")
    channel = cv2.LUT(channel, table)
    return channel


def Summer(img1, val):
    img = img1.copy()
    img[:,:,0] = gamma(img[:,:,0], (1-val))
    img[:,:,2] = gamma(img[:,:,2], (1+val))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = gamma(hsv[:,:,1], 1.15)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def winter(img1, val):
    img = img1.copy()
    img[:,:,0] = gamma(img[:,:,0], (1+val))
    img[:,:,2] = gamma(img[:,:,2], (1-val))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = gamma(hsv[:,:,1], 0.85)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def sepia(img):
    img = np.array(img, dtype=np.float64)
    img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
                                        [0.349, 0.686, 0.168],
                                        [0.393, 0.769, 0.189]]))
    img[np.where(img>255)] = 255
    img = np.array(img, dtype=np.uint8)
    return img


def grain(img, val):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(h):
        for j in range(w):
            if np.random.rand() <= val:
                if np.random.randint(2) == 0:
                    gray[i, j] = min(gray[i, j] + np.random.randint(0, 64), 255)
                else:
                    gray[i, j] = max(gray[i, j] - np.random.randint(0, 64), 0)
    return gray


def pencil_gray(img, flag):
    gray, color = cv2.pencilSketch(img, sigma_r=0.1, sigma_s=100, shade_factor=0.05)
    if flag:
        return color
    return gray


def youtube_import():
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
    return values[0]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989*r + 0.587*g + 0.114*b
    return gray
