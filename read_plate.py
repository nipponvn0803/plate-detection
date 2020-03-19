import cv2
import numpy as np
import argparse
from YOLO import detect
from keras.models import model_from_json
from os.path import splitext

def im2single(Image):
    return Image.astype('float32') / 255

def load_model(path):
    path = splitext(path)[0]
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights('%s.h5' % path)
    return model

# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                    key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Đường dẫn ảnh
img_path = "C:/Users/vuong/Desktop/new1/Detection_LP/test011.jpg"

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)

# dung yolo de detect bien so
ap = argparse.ArgumentParser()
args = ap.parse_args()
YoloImg = detect(args,img_path)

# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')

LpImg=im2single(YoloImg)

if (len(LpImg)):

# =============================================================================
#     # Chuyen doi anh bien so
#     LpImg = cv2.convertScaleAbs(LpImg, alpha=(255.0))
#     roi = LpImg
#     # Chuyen anh bien so ve gray
#     LpImg 
#     LpImg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
#     gray = cv2.cvtColor( LpImg, cv2.COLOR_BGR2GRAY)
#     
#     # Ap dung threshold de phan tach so va nen
#     binary = cv2.threshold(gray, 127, 255,
#                          cv2.THRESH_BINARY_INV)[1]
# 
# =============================================================================
      
    roi = cv2.convertScaleAbs(LpImg, alpha=(255.0))
    
    gray = cv2.cvtColor(YoloImg, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    binary = cv2.threshold(binary, 127, 255,
                         cv2.THRESH_BINARY_INV)[1]
    


    cv2.imshow("Anh bien so sau threshold", binary)
    cv2.waitKey()

    # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    plate_info = ""

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
            if h/roi.shape[0]>=0.2: # Chon cac contour cao tu 60% bien so tro len

                # Ve khung chu nhat quanh so
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Tach so va predict
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                curr_num = np.array(curr_num,dtype=np.float32)
                curr_num = curr_num.reshape(-1, digit_w * digit_h)

                # Dua vao model SVM
                result = model_svm.predict(curr_num)[1]
                result = int(result[0, 0])

                if result<=9: # Neu la so thi hien thi luon
                    result = str(result)
                else: #Neu la chu thi chuyen bang ASCII
                    result = chr(result)

                plate_info +=result

    cv2.imshow("Cac contour tim duoc", roi)
    cv2.waitKey()

    # Viet bien so len anh
    cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

    # Hien thi anh
    print("Bien so=", plate_info)
    cv2.imshow("Hinh anh output",Ivehicle)
    cv2.waitKey()

cv2.destroyAllWindows()