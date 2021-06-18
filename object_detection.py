import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import cvlib as cv
from cvlib.object_detection import draw_bbox

model = tf.keras.models.model_from_json(
    open("/home/rants/PycharmProjects/kbs-project/json_files/object_detection.json", "r").read())

# loading the weights
model.load_weights('/home/rants/PycharmProjects/kbs-project/models/object_detection.h5')

objects = []
locations = []


# taking in my video input function
def video_to_images(video):
    cap = cv2.VideoCapture(video)
    success, image = cap.read()
    count = 0
    while success:
        cv2.imwrite("/home/rants/PycharmProjects/kbs-project/source/flask_app/static/frames/frame%d.jpg" % count,
                    image)  # save frame as JPEG file
        success, image = cap.read()
        if not success:
            print('Done')
        count += 1


def search(list, item):
    for i in range(len(list)):
        if list[i] == item:
            return True
    return False


def detect_objects(folder):
    count = 0
    for file in os.listdir(folder):
        if count <= 10:
            full_path = 'static/frames/' + file
            image = load_img(full_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            y_pred = model.predict(image)
            labels = decode_predictions(y_pred, top=1)
            im = cv2.imread(full_path)
            bbox, label, conf = cv.detect_common_objects(im)
            output_image = draw_bbox(im, bbox, label, conf)
            for i in labels[0][0]:
                if not search(objects, labels[0][0][1]):
                    objects.append(labels[0][0][1])

            for i in label:
                if not search(objects, i):
                    objects.append(i)

            locations.append(full_path)
            count += 1
            print('Done')
        else:
            break


def search_object(item):
    if search(objects, item):
        temp = objects.index(item)
        temp_location = locations[temp]
        im = cv2.imread(temp_location)
        bbox, label, conf = cv.detect_common_objects(im)
        output_image = draw_bbox(im, bbox, label, conf)
        # output_frame(output_image)
        print("The object: %s is found in this frame of the video" % item)
        # print(temp_location)
        return temp_location
