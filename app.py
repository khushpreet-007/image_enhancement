import cv2
import math
import numpy as np
import streamlit as st
from PIL import Image, ImageCms
# import plotly.figure_factory as ff

st.title('Retina Image Enhancement')
uploaded_image = st.file_uploader("Choose an Image ...", type="jpg")



def display_img(img, cap):
    st.image(img,caption=cap)

# def display_chart(img, heading):
#     st.text(heading)
#     hist = cv2.calcHist([img],[0],None,[256],[0,256])
#     st.bar_chart(hist)

def gamma_corrected_fun(img):
    img = img.convert('HSV')
    H, S, V = img.split()
    mid = 0.5
    mean = np.mean(V)
    gamma = math.log(mid*255)/math.log(mean)
    # print(gamma)       
    val_gamma = np.power(V, gamma).clip(0,255).astype(np.uint8)
    H_local = np.array(H)
    S_local = np.array(S)
    V_local = np.array(val_gamma)
    gamma_corrected = cv2.merge([H_local, S_local, V_local])
    img_gamma2 = cv2.cvtColor(gamma_corrected, cv2.COLOR_HSV2BGR)
    return img_gamma2

def clahe_fun(img):
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(img, rgb2lab)
    # C:\Users\Dell\Desktop\ml project
    L, a, b = Lab.split()
    L_local = np.array(L)
    a_local = np.array(a)
    b_local = np.array(b)

    clahe = cv2.createCLAHE()
    V_clahe_img = clahe.apply(L_local)
    V_clahe_img= cv2.merge([V_clahe_img, a_local, b_local])

    output_img = cv2.cvtColor(V_clahe_img, cv2.COLOR_LAB2BGR)
    return output_img         

def imageEnhancement(): 
    if uploaded_image is not None:
        # Luminosity (Gamma correction)
        img = Image.open(uploaded_image)
        gamma_corrected_image = gamma_corrected_fun(img)

        # Contrast (Clahe)
        img = Image.fromarray(gamma_corrected_image)
        output_img = clahe_fun(img)
       
        #  Display Image
        display_image = Image.open(uploaded_image)
        display_img(display_image,"Image Uploaded from user")
        display_img(output_img, "Image after clahe")
         
        #  Display Bar chart
        # display_chart(gamma_corrected_image, "Bar Chart Before Clahe Algorithm")
        # display_chart(output_img, "Bar Chart after Clahe Algorithm")

if __name__ == '__main__':
    imageEnhancement() 

