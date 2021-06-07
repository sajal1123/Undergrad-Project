import cv2
import pafy
import youtube_dl
import youtubedownloader
import youtube_dl_cli

url = "https://www.youtube.com/watch?v=W_zmPMXQPAc"
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")

cap = cv2.VideoCapture(play.url)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()