import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
from tensorflow.keras.models import model_from_json

# Load model
with open('model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.weights.h5")

# Character mapping
mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '-', 11: '+', 12: '*',13:'/',14:'x',15:'y'}

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/solve', methods=['POST'])
def solve_equation():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    img = ~img
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    train_data = []
    rects = [cv2.boundingRect(c) for c in cnt]
    bool_rect = [[1 if (r != rec and r[0] < rec[0] + rec[2] + 10 and rec[0] < r[0] + r[2] + 10 and r[1] < rec[1] + rec[3] + 10 and rec[1] < r[1] + r[3] + 10) else 0 for rec in rects] for r in rects]
    dump_rect = [rects[i] for i in range(len(cnt)) for j in range(len(cnt)) if bool_rect[i][j] == 1 and rects[i][2] * rects[i][3] <= rects[j][2] * rects[j][3]]
    final_rect = [r for r in rects if r not in dump_rect]

    for r in final_rect:
        x, y, w, h = r
        im_crop = thresh[y:y + h + 10, x:x + w + 10]
        im_resize = cv2.resize(im_crop, (28, 28))
        im_resize = np.reshape(im_resize, (1, 28, 28, 1))
        im_resize = im_resize.astype("float32") / 255.0
        train_data.append(im_resize)

    s = ''
    for i in range(len(train_data)):
        result = np.argmax(loaded_model.predict(train_data[i], verbose=0))
        s += mapping[result]

    # Evaluate the equation
    try:
        evaluation = eval(s)
    except:
        evaluation = "Invalid Expression"

    return jsonify({"expression": s, "result": evaluation})

if __name__ == "__main__":
    app.run(debug=True)