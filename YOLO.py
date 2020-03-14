import time
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()

ap.add_argument('-c', '--config', default='yolov3.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default='yolov3.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='yolo.names',
                help='path to text file containing class names')
args = ap.parse_args()
classes = None
with open(args.classes, 'r') as f:
  classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
net = cv2.dnn.readNet(args.weights, args.config)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def detect(args,imagepath):
    image=cv2.imread(imagepath)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    
    outs = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    # Thực hiện xác định bằng HOG và SVM
    start = time.time()
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    if not boxes:
        print("noconfidence")
    else:    
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
 
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    
        croppedImage = image[int(y):int(y+h),int(x):int(x+w)]    
        
        if 0< w/h<= 2.5:
            croppedImage = cv2.resize(croppedImage,(200,200))
        else:
            croppedImage = cv2.resize(croppedImage,(800,200))
    if "croppedImage" in locals():
        
        cv2.imshow("object detection1",image)
        cv2.waitKey()
        
        cv2.imshow("license plate detection", croppedImage)
        end = time.time()
        print("YOLO Execution time: " + str(end-start))
       
        cv2.waitKey()
        cv2.imwrite("license plate detection.jpg", croppedImage)
        cv2.destroyAllWindows()
        
        return croppedImage
    else:
        print("no plate detected")
        
if __name__ == "__main__":
    while True:
        path=input("path = ")
        if(path=="X") :
                break
        detect(args,path)
    