import os
from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "captcha images"

def captcha_solver():
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)
    model = load_model(MODEL_FILENAME)

    captcha_image_files = []
    captcha_name = "E:\SAMPLES\GenerateCaptcha.jpg"
    captcha_image_files.append(captcha_name)

    s_time=time.time()
    for image_file in captcha_image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]

        letter_image_regions = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w / h < 1.07:
                letter_image_regions.append((x, y, w, h))

        if len(letter_image_regions) != 4:
            continue


        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
        output = cv2.merge([image] * 3)
        predictions = []

        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
            letter_image = resize_to_fit(letter_image, 20, 20)
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            prediction = model.predict(letter_image)
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

        captcha_text = "".join(predictions)
        return captcha_text
    print (time.time()-s_time)


    """f= open("captcha_awnser.txt","w")
    f.write(captcha_text)
    f.close()"""
print(captcha_solver())