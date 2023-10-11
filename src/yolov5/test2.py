from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2


img = "data/images/image_8.jpg"
cv2_image = cv2.imread(img)

# Load a model
model = YOLO('yolov8m-seg.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# results = model(['im1.jpg', 'im2.jpg'])  # return a list of Results objects
results = model(cv2_image)  
# Process results list
for result in results:
    # boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    img_array = result.plot()

cv2.imshow('img', img_array)
cv2.waitKey(0)

mask_result = masks.data.cpu().detach().numpy()
mask2 = np.squeeze(mask_result)
mask3 = cv2.resize(mask2, (1920, 1080), interpolation = cv2.INTER_LINEAR)
plt.imshow(mask3, cmap=plt.cm.gray)
plt.show()
