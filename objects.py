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


# im = cv2.imread('/home/rants/PycharmProjects/kbs-project/apples-732x549-thumbnail.jpg')
# bbox, label, conf = cv.detect_common_objects(im)
# output_image = draw_bbox(im, bbox, label, conf)
#
# # plt.imshow(output_image)
# # plt.show()
#
# cv2.imshow("Test", output_image)


def output_frame(output_image):
    while True:
        test_image = cv2.resize(output_image, (224, 224), interpolation= cv2.INTER_AREA)
        cv2.imshow('frame', test_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def search(list, item):
    for i in range(len(list)):
        if list[i] == item:
            return True
    return False


count = 0
objects = []
locations = []
for file in os.listdir('static/frames'):
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
        plt.imshow(output_image)
        for i in labels[0][0]:
            if not search(objects, labels[0][0][1]):
                objects.append(labels[0][0][1])

        for i in label:
            if not search(objects, i):
                objects.append(i)

        locations.append(full_path)
        # print(labels[0][0][1])
        # print()
        count += 1
    else:
        break


def searchObject(item):
    if search(objects, item):
        temp = objects.index(item)
        temp_location = locations[temp]
        locations[temp]
        im = cv2.imread(temp_location)
        bbox, label, conf = cv.detect_common_objects(im)
        output_image = draw_bbox(im, bbox, label, conf)
        output_frame(output_image)
        print("The object: %s is found in this frame of the video" % item)


print(objects)
print()
print(searchObject('chair'))
