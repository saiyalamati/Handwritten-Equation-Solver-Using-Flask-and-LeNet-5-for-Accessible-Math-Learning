import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

# Load model
with open('model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.weights.h5")

# Load and preprocess image
img = cv2.imread('Test Image.jpeg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Error: Image not found or unable to read.")

img = ~img  # Invert colors
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours_info = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours_info[0] if len(contours_info) == 2 else contours_info[1]

# Extract bounding boxes
rects = [cv2.boundingRect(c) for c in sorted(cnt, key=lambda ctr: cv2.boundingRect(ctr)[0])]

# Merge overlapping bounding boxes
bool_rect = [
    [1 if (r != rec and r[0] < rec[0] + rec[2] + 10 and rec[0] < r[0] + r[2] + 10 and
            r[1] < rec[1] + rec[3] + 10 and rec[1] < r[1] + r[3] + 10) else 0
     for rec in rects] for r in rects
]

dump_rect = [rects[i] for i in range(len(cnt)) for j in range(len(cnt))
             if bool_rect[i][j] == 1 and rects[i][2] * rects[i][3] <= rects[j][2] * rects[j][3]]

final_rect = [r for r in rects if r not in dump_rect]

# Process each detected symbol
train_data = []
for r in final_rect:
    x, y, w, h = r
    im_crop = thresh[y:y + h + 10, x:x + w + 10]
    im_resize = cv2.resize(im_crop, (28, 28))
    im_resize = np.reshape(im_resize, (1, 28, 28, 1))  # ✅ Fix input shape
    im_resize = im_resize.astype("float32") / 255.0  # Normalize
    train_data.append(im_resize)

# Mapping of output labels
mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
           8: '8', 9: '9', 10: '-', 11: '+', 12: '*'}

s = ''
for i in range(len(train_data)):
    result = np.argmax(loaded_model.predict(train_data[i], verbose=0))  # Fix prediction handling
    s += mapping[result]

print("\nThe evaluation of the image gives equation:", s, "\n")

# Evaluate the equation safely
try:
    print("\nThe evaluation of the image gives -->", s, "=", eval(s), "\n")
except:
    print("\nError in evaluating the equation. Invalid expression:", s, "\n")
