import cv2
import cvzone
from cvzone.ClassificationModule import Classifier
import os

# ... (Your existing code for setup)
cap = cv2.VideoCapture(0)

classifier = Classifier(r'C:\Users\Dell\Desktop\CAPSTONE-PROJECT\keras_model.h5', r'C:\Users\Dell\Desktop\CAPSTONE-PROJECT\labels.txt')
imgArrow = cv2.imread(r'C:\Users\Dell\Desktop\CAPSTONE-PROJECT\arrow.png', cv2.IMREAD_UNCHANGED)
classIDBin = 0
# Import all the waste images
imgWasteList = []
pathFolderWaste = r'C:\Users\Dell\Desktop\CAPSTONE-PROJECT\Waste'
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

imgBinsList = []
pathFolderBins = r'C:\Users\Dell\Desktop\CAPSTONE-PROJECT\Bins'
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))
classDic = {0: 0, 1: 1}
while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))

    imgBackground = cv2.imread(r'C:\Users\Dell\Desktop\CAPSTONE-PROJECT\background.png')

    prediction = classifier.getPrediction(img)
    classID = prediction[1]
    if classID != -1:
        # The bounding box coordinates around the detected waste
        x, y, w, h = 100, 100, 150, 150 # Change these values according to your detected waste location
        # Draw bounding box on the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
        classIDBin = classDic[classID]
        imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    imgBackground[148:148 + 340, 159:159 + 454] = imgResize
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)