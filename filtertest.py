import time
import cv2 # For OpenCV modules (For Image I/O and Contour Finding)
import numpy as np # For general purpose array manipulation
import scipy.fftpack # For FFT2 
from lib_detection import load_model, detect_lp, im2single
import PIL

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

# imclearborder definition

# =============================================================================
# def imclearborder(imgBW, radius):
# 
#     # Given a black and white image, first find all of its contours
#     imgBWcopy = imgBW.copy()
#     contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
#         cv2.CHAIN_APPROX_SIMPLE)
# 
#     # Get dimensions of image
#     imgRows = imgBW.shape[0]
#     imgCols = imgBW.shape[1]    
# 
#     contourList = [] # ID list of contours that touch the border
# 
#     # For each contour...
#     for idx in np.arange(len(contours)):
#         # Get the i'th contour
#         cnt = contours[idx]
# 
#         # Look at each point in the contour
#         for pt in cnt:
#             rowCnt = pt[0][1]
#             colCnt = pt[0][0]
# 
#             # If this is within the radius of the border
#             # this contour goes bye bye!
#             check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
#             check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)
# 
#             if check1 or check2:
#                 contourList.append(idx)
#                 break
# 
#     for idx in contourList:
#         cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)
# 
#     return imgBWcopy
# =============================================================================

# bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy



# Đường dẫn ảnh
img_path = "C:/Users/vuong/Desktop/new1/Detection_LP/test27.jpg"

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu


model_svm = cv2.ml.SVM_load('svm.xml')


# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

start = time.time()

_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

end = time.time()
print("Detection Execution time: " + str(end-start))

if (len(LpImg)):

    # Chuyen doi anh bien so
    roi = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    cv2.imshow("anh bien so detect",roi)
    cv2.waitKey()   
    
    
    if 0.7 < roi.shape[0]/roi.shape[1] < 1.5:
        image = roi

        cv2.imshow("Original Image", image)
        cv2.waitKey() 
        
        height, width = image.shape[:2]
        
            
        # Let's get the starting pixel coordiantes (top left of cropped top)
        start_row, start_col = int(0), int(0)
        # Let's get the ending pixel coordinates (bottom right of cropped top)
        end_row, end_col = int(height * .5), int(width)
        cropped_top = image[start_row:end_row , start_col:end_col]
        print (start_row, end_row) 
        print (start_col, end_col)
        
        cv2.imshow("Cropped Top", cropped_top) 
        img1 = cropped_top
        cv2.waitKey() 
        cv2.destroyAllWindows()
        
        # Let's get the starting pixel coordiantes (top left of cropped bottom)
        start_row, start_col = int(height * .5), int(0)
        # Let's get the ending pixel coordinates (bottom right of cropped bottom)
        end_row, end_col = int(height), int(width)
        cropped_bot = image[start_row:end_row , start_col:end_col]
        print (start_row, end_row) 
        print (start_col, end_col)
        
        img2 = cropped_bot
        cv2.imshow("Cropped Bot", cropped_bot) 
        cv2.waitKey()    
        
        img1 = PIL.Image.fromarray(img1)
        img2 = PIL.Image.fromarray(img2)
        
        img1.save("top.jpg")
        img2.save("bottom.jpg")
        
        list_im = ["top.jpg","bottom.jpg"]
        imgs    = [ PIL.Image.open(i) for i in list_im ]
        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
        
        # save that beautiful picture
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save( 'square_sorted.jpg' )
                
        plateImg = cv2.imread('square_sorted.jpg')
        roi = cv2.convertScaleAbs(plateImg, alpha=(255.0))
        gray = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)
      
        
    else:    
        # Chuyen anh bien so ve gray
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
     
        cv2.waitKey()
   
    # Main program
    # Read in image
    img = gray
    cv2.imshow("12d",img)
    cv2.waitKey()
    
    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]
    
    
    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)
    
    # Create Gaussian mask of sigma = 10
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 10
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
    
    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow
    
    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
    
    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = np.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = np.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))
    
    # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]
    
    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")
    
    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2 < 65
    Ithresh = 255*Ithresh.astype("uint8")
    
# =============================================================================
#     # Clear off the border.  Choose a border radius of 5 pixels
#     Iclear = imclearborder(Ithresh, 1)
# =============================================================================
    
    # Eliminate regions that have areas below 120 pixels
    Iopen = bwareaopen(Ithresh, 120)
    
    # Show all images
    cv2.imshow('Original Image', img)
    cv2.imshow('Homomorphic Filtered Result', Ihmf2)
    cv2.imshow('Thresholded Result', Ithresh)
    cv2.imshow('Opened Result', Iopen)
    cv2.waitKey()
    
  
    # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(Iopen, cv2.MORPH_DILATE, kernel3)
    cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    plate_info = ""
        
    for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                if h/roi.shape[0]>=0.68: # Chon cac contour cao tu 60% bien so tro len
    
                    # Ve khung chu nhat quanh so
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
                    # Tach so va predict
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
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
    
    cv2.imshow("Cac contour tim duoc", img)
    cv2.waitKey()

    # Viet bien so len anh
    cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

    # Hien thi anh
    print("Bien so=", plate_info)
    cv2.imshow("Hinh anh output",Ivehicle)
    cv2.waitKey()



cv2.destroyAllWindows()
