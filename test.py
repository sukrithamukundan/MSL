from keras.models import load_model
import cv2
import numpy as np
import os

IMG_TEST_PATH = "test_image"

label_lst = ['A','Aa','E','Ee']
NUM_CLASSES = len(label_lst)
REV_CLASS_MAP = {i:label_lst[i] for i in range(NUM_CLASSES)}

def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("malayalam-sign-language-model.h5")

for directory in os.listdir(IMG_TEST_PATH):
    path = os.path.join(IMG_TEST_PATH,directory)
    if not os.path.isdir(path):
        continue
    
    for file_name in os.listdir(path):
        # to make sure no hidden files get in our way
        if file_name.startswith("."):
            continue
        # to make sure only jpg is extracted
        if not file_name.endswith(".jpg"):
            continue

        image_path = os.path.join(path,file_name)
        # prepare the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))

        # predict the sign made
        pred = model.predict(np.array([img]))
        sign_code = np.argmax(pred[0])
        sign_name = mapper(sign_code)

        print(f"Predicted: {sign_name} for image {image_path}")
