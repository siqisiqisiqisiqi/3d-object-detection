from ultralytics import YOLO
import cv2


img = "data/images/image_8.jpg"
cv2_image = cv2.imread(img)
def draw_box(cv2_image, xmin, xmax, ymin, ymax):
        # top
        cv2.line(cv2_image, (xmin, ymax), (xmax, ymax), (0, 255, 0), 4)
        # bottom
        cv2.line(cv2_image, (xmin, ymin), (xmax, ymin), (0, 255, 0), 4)
        # left
        cv2.line(cv2_image, (xmin, ymin), (xmin, ymax), (0, 255, 0), 4)
        # right
        cv2.line(cv2_image, (xmax, ymin), (xmax, ymax), (0, 255, 0), 4)
        # center
        cv2.circle(cv2_image, (int((xmin+xmax)/2), int((ymin+ymax)/2)), 5, (0,0,255), 5)

# Load a model
model = YOLO('yolov8s.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# results = model(['im1.jpg', 'im2.jpg'])  # return a list of Results objects
results = model(img)  
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

cords = boxes.xyxy[0].tolist()
categ = boxes.cls.cpu().detach().numpy().astype('int')

coords = []
for i in range(len(categ)):
    if categ[i] == 64:
        coords.append(boxes.xyxy[i].tolist())

for coord in coords:
    xmin = int(coord[0])
    ymin = int(coord[1])
    xmax = int(coord[2])
    ymax = int(coord[3])

draw_box(cv2_image, xmin, xmax, ymin, ymax)
cv2.imshow('img', cv2_image)
cv2.waitKey(0)