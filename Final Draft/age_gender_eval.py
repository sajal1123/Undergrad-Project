
import numpy as np
import argparse
import cv2
import os
import cv2.dnn
import matplotlib.pyplot as plt
import glob2 as glob

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



ctr=0
AGE_BUCKETS = ["(0-2)", "(4-6)", "(6-12)", "(13-20)", "(21-30)",
	"(31-40)", "(41-50)", "(51-60)", "(60-100)"]
age_list = []
age_real = []
gender_list_1 = []
gender_real = []
print("Started")
for image in glob.iglob('C:/Users/Sajal/PycharmProjects/Minor_Project/crop_part1/*.jpg'):
    temp = image.split('\\')[0] + '/' + image.split('\\')[1]
    # print(temp)
    ctr += 1
    # print("ctr=",ctr)
    cap = cv2.imread(temp)
    if ctr%123 == 0:
        # cv2.imshow("AI", cap)
        # cv2.waitKey(0)
        # print(cap)
        #(h, w) = cap.shape[:2]
        blob = cv2.dnn.blobFromImage(cap, 1.0, (300, 300), (104.0, 177.0, 123.0))

                # pass the blob through the network and obtain the face detections
                #print("[INFO] computing face detections...")
        faceNet.setInput(blob)
        detections = faceNet.forward()
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale3(gray, scaleFactor=1.08, minNeighbors=5,
                                           minSize=(30, 30), outputRejectLevels=True)

        if len(faces) > 0:
            rects = faces[0]
            for (x, y, w, h) in rects:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 200, 0), 3)

            # loop over the detections
            for i in range(0, detections.shape[2]):
                            # extract the confidence (i.e., probability) associated with the
                            # prediction
                confidence = detections[0, 0, i, 2]

                        # filter out weak detections by ensuring the confidence is
                        # greater than the minimum confidence
                if confidence > 0.5:
                            # compute the (x, y)-coordinates of the bounding box for the
                            # object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                            # extract the ROI of the face and then construct a blob from
                            # *only* the face ROI
                    face = cap
                    try:
                        ageBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                                    (78.4263377603, 87.7689143744, 114.895847746),
                                                                    swapRB=False)

                        genderBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                                       (78.4263377603, 87.7689143744, 114.895847746),
                                                                       swapRB=False)
                    except:
                        # print("FAILED : NO BLOBS FOUND")
                        pass

                                # make predictions on the age and find the age bucket with
                                # the largest corresponding probability
                    try:
                        ageNet.setInput(ageBlob)
                        preds = ageNet.forward()
                        # print("PREDICTION -> ", preds)
                        i = preds[0].argmax()
                        # print("AGE -> ",i)
                        age = AGE_BUCKETS[i]
                        ageConfidence = preds[0][i]
                        age_list.append(age)
                        age_real.append(int(temp.split("/")[-1].split('_')[0]))
                        # print("Predicted Age ->", age, "\tReal Age ->", age_real[-1])
                                    #detecting gender
                        gender_net.setInput(genderBlob)
                        gender_preds = gender_net.forward()
                        gender = gender_list[gender_preds[0].argmax()]
                        gender_list_1.append(gender)
                        if int(temp.split("/")[-1].split('_')[1])==0:
                            gender_real.append('Male')
                        else:
                            gender_real.append("Female")
                        # print("Predicted Gender ->", gender, "\tReal Gender ->", gender_real[-1])

                        if ctr == 5535-123:
                            cv2.imshow(age, cap)
                            print(image)
                            print(age)
                            print(gender)
                            cv2.waitKey(0)

                    except:
                        # print("FAILED : NO AGE GENDER DETECTED")
                        pass

# print("PredAge -> ", age_list)
# print("RealAge -> ", age_real)
# print("PredGender -> ", gender_list_1)
# print("RealGender -> ", gender_real)

age_accuracy = []
# gender_accuracy = []
age_buckets_correct = {}
age_buckets_total = {}
for item in AGE_BUCKETS:
    age_buckets_correct[item] = 0
    age_buckets_total[item] = 0
for j in range(0, 11):
    correct = 0
    total = 0
    for i in range(len(age_list)):
        asli = age_real[i]
        # tolerance = asli/2
        if (asli - int(age_list[i].split('-')[0][1:])) > -j and (asli < int(age_list[i].split('-')[1][:-1])) < j:
            correct += 1
            age_buckets_correct[age_list[i]] += 1
        total += 1
        age_buckets_total[age_list[i]] += 1
    age_accuracy.append((100*correct/total))
    # print("For J = ",j,"  Correct Age Predictions -> ", correct, " out of ", total, "\t Accuracy = ",(100*correct/total),"%")

correct = 0
total = 0
for i in range(len(gender_list_1)):
    if gender_list_1[i] == gender_real[i]:
        correct += 1
    total += 1
# print("Correct Gender Predictions -> ", correct, " out of ", total, "\t Accuracy = ",(100*correct/total),"%")
# plt.plot(np.arange(0, 11), age_accuracy)
# plt.xlabel('Tolerance level(yrs)')
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy v/s Tolerance")
# plt.show()
# # print("AGE BUCKETS TOTAL ->", age_buckets_total)
# # print("AGE BUCKETS CORRECT ->", age_buckets_correct)
# # age_buckets_accuracy = {}
# # age_buckets_correct["(13-20)"] += 301
# # age_buckets_total["(13-20)"] += 410
# # age_buckets_correct["(60-100)"] += 329
# # age_buckets_total["(60-100)"] += 386
# for item in AGE_BUCKETS:
#     try:
#         age_buckets_accuracy[item] = age_buckets_correct[item]/age_buckets_total[item]
#     except:
#         pass
# print("AGE BUCKETS ACCURACY ->", age_buckets_accuracy)
#
# plt.bar(list(age_buckets_accuracy.keys()), list(age_buckets_accuracy.values()))
# plt.xlabel("AGE BUCKETS")
# plt.ylabel("PREDICTION ACCURACY")
# plt.title("Model Accuracy for each age group")
# plt.show()