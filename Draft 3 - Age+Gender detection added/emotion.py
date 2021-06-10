from PIL import Image
from mtcnn import MTCNN
import pickle
from functions import *

detector = MTCNN()

sex_model = pickle.load(open('C:/Users/Sajal/PycharmProjects/Minor_Project/models/sex-model-final.pkl', 'rb'))
age_model = pickle.load(open('C:/Users/Sajal/PycharmProjects/Minor_Project/models/age-model-final.pkl', 'rb'))
emotion_model = pickle.load(open('C:/Users/Sajal/PycharmProjects/Minor_Project/models/emotion-model-final.pkl', 'rb'))

sex_model._get_distribution_strategy = lambda: None
age_model._get_distribution_strategy = lambda: None
emotion_model._get_distribution_strategy = lambda: None

def detect_face(img):
    mt_res = detector.detect_faces(img)
    return_res = []

    for face in mt_res:
        x, y, width, height = face['box']
        center = [x+(width//2), y+(height//2)]
        max_border = max(width, height)

        # center alignment
        left = max(int(center[0]-(max_border/2)), 0)
        right = max(int(center[0]+(max_border/2)), 0)
        top = max(int(center[1]-(max_border/2)), 0)
        bottom = max(int(center[1]+(max_border/2)), 0)

        # crop the face
        center_img_k = img[top:top+max_border,
                       left:left+max_border, :]
        center_img = np.array(Image.fromarray(center_img_k).resize([224, 224]))

        # create predictions
        sex_preds = sex_model.predict(center_img.reshape(1,224,224,3))[0][0]
        age_preds = age_model.predict(center_img.reshape(1,224,224,3))[0][0]

        # convert to grey scale then predict using the emotion model
        grey_img = np.array(Image.fromarray(center_img_k).resize([48, 48]))
        emotion_preds = emotion_model.predict(rgb2gray(grey_img).reshape(1, 48, 48, 1))

        # output to the cv2
        return_res.append([top, right, bottom, left, sex_preds, age_preds, emotion_preds])

    return return_res

emotion_dict = {
    0:'Surprise',
    1:'Happy',
    2:'Disgust',
    3:'Anger',
    4:'Sadness',
    5:'Fear',
    6:'Contempt'
}



