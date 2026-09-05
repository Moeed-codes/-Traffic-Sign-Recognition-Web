import os
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import google.generativeai as genai
import traceback

#  CONFIG 
APP_NAME = "Leaf Ray Net Detection"
MODEL_PATH = "plant_disease_model.h5"      
JSON_PATH = "class_indices.json"           
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API CONFIGURATION
# API CONFIGURATION
# Load environment variables manually to avoid dependency on python-dotenv
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
        
    genai.configure(api_key=GEMINI_API_KEY)
    # User requested Flash model
    model_api = genai.GenerativeModel('gemini-2.5-flash')
    API_AVAILABLE = True
except Exception as e:
    print(f"API Setup Error: {e}")
    API_AVAILABLE = False

#  APP 
app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB limit

#  TREATMENT LOGIC 
def get_treatment_info(disease_name):
    """
    Fetches treatment info from Gemini API.
    Returns a list of strings (bullet points).
    """
    if not API_AVAILABLE:
        return ["API Key not configured. Please add your Gemini API Key in app.py to see AI-generated treatments."]
    
    if "healthy" in disease_name.lower():
        return ["Plant is healthy. No treatment needed.", "Maintain regular care."]

    prompt = f"Provide 3-4 short, actionable bullet points on how to cure or treat the plant disease '{disease_name}'. Do not include introductory text, just the bullet points."
    
    try:
        response = model_api.generate_content(prompt)
        text = response.text
        # Clean up response to get a nice list
        lines = [line.strip().lstrip('-•* ') for line in text.split('\n') if line.strip()]
        return lines[:5] # Limit to 5 points
    except Exception as e:
        print(f"API Generation Error: {e}")
        return ["Error fetching treatments from AI. Please try again later."]

#  HELPERS 
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

#  LOAD MODEL 
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Force build by running a dummy prediction
try:
    print("Initializing model with dummy input...")
    dummy_input = np.zeros((1, 150, 150, 3))
    model.predict(dummy_input)
    print("Model initialized.")
except Exception as e:
    print(f"Model initialization warning: {e}")

print("Model loaded:", MODEL_PATH)

with open(JSON_PATH, "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# GRAD-CAM & CIRCLE LOGIC 
def find_last_conv_layer(model):
    """
    Recursively find the last 4D Convolutional layer.
    """
    for layer in reversed(model.layers):
        # Check if it's a Conv2D layer by name or type
        if "conv" in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
            try:
                # Verify it has 4D output just in case
                if hasattr(layer, 'output_shape'):
                    shape = layer.output_shape
                elif hasattr(layer, 'get_output_shape_at'):
                    shape = layer.get_output_shape_at(0)
                else:
                    # If we can't get shape but it's a conv layer, assume it's good
                    return layer.name
                
                if len(shape) == 4:
                    return layer.name
            except:
                # If shape check fails but it's a conv layer, return it
                return layer.name
        
        # If it's a nested model (Transfer Learning), dig inside
        if hasattr(layer, 'layers'):
            deep_layer = find_last_conv_layer(layer)
            if deep_layer:
                return deep_layer
    return None

def process_disease_visualization(img_path, model, gradcam_path, circle_path, input_size=(150,150)):
    try:
        # 1. Preprocess
        img = image.load_img(img_path, target_size=input_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        # 2. Deep Search for Layer
        last_conv_name = find_last_conv_layer(model)
        if not last_conv_name:
            print("Error: No 4D Conv layer found in model.")
            return False, False
            
        print(f"Grad-CAM using layer: {last_conv_name}")

        # 3. Create Grad-Model
        # Rebuild graph explicitly to ensure connectivity
        print("DEBUG: Rebuilding graph for Grad-CAM...")
        grad_model_input = tf.keras.Input(shape=(150, 150, 3))
        x_tensor = grad_model_input
        conv_output = None
        
        # Iterate through layers and apply them to the new input
        for layer in model.layers:
            x_tensor = layer(x_tensor)
            if layer.name == last_conv_name:
                conv_output = x_tensor
        
        if conv_output is None:
            print(f"Error: Layer {last_conv_name} not found in model traversal.")
            return False, False

        grad_model = tf.keras.models.Model(grad_model_input, [conv_output, x_tensor])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.sum(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        # 4. Save Heatmap & Draw Circle
        orig = cv2.imread(img_path)
        heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Force Circle: Find the absolute brightest spot (Max Loc)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(heatmap_uint8)
        
        circle_img = orig.copy()
        
        # Draw Red Circle at the "hottest" point
        cv2.circle(circle_img, maxLoc, 60, (0, 0, 255), 4) # Radius 60, Red
        cv2.circle(circle_img, maxLoc, 5, (0, 0, 255), -1) # Center dot
        
        cv2.imwrite(circle_path, circle_img)
        
        # Save GradCAM just in case
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(heatmap_colored, 0.5, orig, 0.5, 0)
        cv2.imwrite(gradcam_path, superimposed)

        return True, True

    except Exception as e:
        print("Visualization Error:", e)
        traceback.print_exc()
        with open("viz_error.log", "w") as f:
            f.write(f"Error: {str(e)}\n")
            traceback.print_exc(file=f)
        return False, False

def predict_image(img_path, model, top_k=5, input_size=(150,150)):
    img = image.load_img(img_path, target_size=input_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = model.predict(x)[0]
    idxs = np.argsort(preds)[::-1][:top_k]
    results = []
    for idx in idxs:
        cls_name = class_names.get(int(idx), "Unknown")
        prob = float(preds[int(idx)])
        results.append({"class_name": cls_name, "probability": prob})
    return results

#  ROUTES 
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", app_name=APP_NAME)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "No selected file or invalid extension"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # 1. Predictions
    topk = predict_image(save_path, model, top_k=5)
    
    readable = []
    for item in topk:
        cls_key = item["class_name"]
        pretty = cls_key.replace("___", " - ").replace("_", " ")
        prob = round(item["probability"] * 100, 2)
        readable.append({"name": pretty, "prob": prob, "raw_name": cls_key})

    # 2. Visualizations (Grad-CAM + Circle)
    grad_name = "gradcam_" + filename
    circle_name = "circle_" + filename
    
    grad_path = os.path.join(app.config["UPLOAD_FOLDER"], grad_name)
    circle_path = os.path.join(app.config["UPLOAD_FOLDER"], circle_name)
    
    got_grad, got_circle = process_disease_visualization(save_path, model, grad_path, circle_path)

    # 3. Treatments
    best_raw = topk[0]["class_name"] if len(topk) > 0 else None
    treatments = get_treatment_info(best_raw) if best_raw else ["No disease detected."]

    response = {
        "predictions": readable,
        "image_url": "/" + save_path.replace("\\", "/"),
        "gradcam_url": ("/" + grad_path.replace("\\", "/")) if got_grad else None,
        "circle_url": ("/" + circle_path.replace("\\", "/")) if got_circle else None,
        "treatments": treatments
    }
    return jsonify(response)

@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)