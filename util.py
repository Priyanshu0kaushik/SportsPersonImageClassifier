import cv2
import pandas as pd
import streamlit as st
import numpy as np
import pickle
from PIL import Image
import json

def classify_img(loaded_img):
    faces = cropped_img(loaded_img)
    final_img = []
    if len(faces) > 0:
        for img in faces:
            resized_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            resized_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((resized_img.reshape(32 * 32 * 3, 1), resized_img_har.reshape(32 * 32, 1)))

            image_array_len = 32*32*3 + 32*32
            final_img.append(combined_img.reshape(1, image_array_len).astype(float))

        load_model()
        result = []
        st.markdown("###")
        for i in range(len(final_img)):
            result.append(model_.predict(final_img[i]))
            show_result(result[i])
            # show_proba(model_.predict_proba(final_img[i]))
        # st.write(result)



        # show_proba(model_.predict_proba(final_img[0]))

    else:
        st.markdown("###")
        a, b, c = st.columns([0.25, 0.6, 0.15])

        with b:
            st.subheader("Eyes Not Detected..\nPlease Try a different Image.")


def load_file(file):
    if file is not None:
        with open('img.png', 'wb') as f:
            f.write(file.read())
        try:
            img = cv2.imread('img.png')
            a, b, c = st.columns([0.15, 0.7, 0.15])

            with b:
                image = Image.open('img.png')
                aspect_ratio = image.width / image.height
                new_width = 300
                new_height = int(new_width / aspect_ratio)
                resized = image.resize((new_width, new_height))
                st.image(resized)

            return img

        except:
            col1, col, col2 = st.columns([0.20, 0.55, 0.15])
            with col:
                st.subheader("Please Upload an Image..")



def cropped_img(img):
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    result = []
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                result.append(roi_color)

        return result

import pywt


def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255

    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def load_model():
    global model_
    global names_dict
    model_ = pickle.load(open('image_model.pickle', 'rb'))
    names_di = json.load(open('names_dict.json'))
    names_dict = {}
    for name in names_di.keys():
        new_name = list(name.split("_"))
        new_name = [a.capitalize() for a in new_name]
        new_name = " ".join(new_name)
        names_dict[new_name] = names_di[name]


def show_result(result):
    for i, j in names_dict.items():
        if j == result:
            name = i
            break

    st.markdown("###")
    a, b, c = st.columns([0.4, 0.6, 0.2])

    with b:
        img = Image.open(f"./imgs/{name}_result.png")
        img = img.resize((200, 200))
        st.image(img)
    col1, col, col2 = st.columns([0.45, 0.33, 0.33])
    with col:
        st.subheader(name)


def show_proba(proba):
    proba = list(np.round(proba[0]*100, 2))
    x = names_dict.copy()
    count = 0
    for i in x:
        x[i] = proba[count]
        count += 1
    key = x.keys()

    proba_list = [x[i] for i in key]

    proba_dict = {
        'Players': key,
        'Probablity (%)': proba_list
    }
    final_df = pd.DataFrame(proba_dict)
    final_df = final_df.sort_values('Probablity (%)', ascending=False)

    a, b, c = st.columns([0.23, 0.60, 0.15])

    with b:
        st.dataframe(final_df, width=350, hide_index=True)


if __name__ == "__main__":
    classify_img(cv2.imread('img.png'))
