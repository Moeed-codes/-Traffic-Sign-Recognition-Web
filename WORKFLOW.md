# Project Workflow: Leaf Ray Net Detection

This document outlines the end-to-end workflow of the Leaf Ray Net Detection application, detailing how the system processes user inputs to detect plant diseases and provide treatment suggestions.

## 1. User Interaction (Frontend)
- **Action**: The user accesses the web interface and uploads an image of a plant leaf.
- **Interface**: The frontend is built with HTML/Bootstrap (`templates/index.html`).
- **Constraint**: The system accepts standard image formats (JPG, PNG) up to 16MB.

## 2. Request Handling (Backend)
- **Server**: A Flask server (`app.py`) receives the POST request containing the image.
- **Validation**: The server checks for a valid file extension and saves the image securely to the `static/uploads` directory.

## 3. Image Preprocessing
- **Resizing**: The image is loaded and resized to **150x150 pixels** to match the neural network's input requirement.
- **Normalization**: Pixel values are normalized (scaled between 0 and 1) to ensure consistent model performance.
- **Batching**: The image is expanded to a batch of size 1 (shape: `1, 150, 150, 3`).

## 4. Disease Detection (CNN Inference)
- **Model**: The application loads a pre-trained TensorFlow/Keras model (`plant_disease_model.h5`).
- **Prediction**: The preprocessed image is passed through the model.
- **Result**: The model outputs probability scores for each of the trained disease classes. The top 5 predictions are extracted.

## 5. Visualization (Grad-CAM & Localization)
- **Grad-CAM**: The system identifies the last convolutional layer of the model and calculates the gradients of the predicted class with respect to the feature maps.
- **Heatmap**: These gradients are used to generate a heatmap that highlights the regions of the image most important for the prediction.
- **Localization**:
    - The heatmap is superimposed on the original image.
    - A **Red Circle** is automatically drawn around the region with the highest activation intensity to pinpoint the disease location.
- **Output**: Two new images are saved: one with the Grad-CAM overlay and one with the localization circle.

## 6. Treatment Recommendation (Generative AI)
- **Condition**: If the predicted class is not "Healthy", the system proceeds to get treatment advice.
- **API Call**: The application connects to **Google's Gemini API** (using the `gemini-2.5-flash` model).
- **Prompt**: A specific prompt is sent: *"Provide 3-4 short, actionable bullet points on how to cure or treat the plant disease '{disease_name}'..."*
- **Response**: The AI generates context-aware treatment steps, which are parsed into a list.

## 7. Response & Display
- **JSON Response**: The backend bundles the following data into a JSON response:
    - Top predictions with confidence scores.
    - URL to the original uploaded image.
    - URL to the Grad-CAM/Localization images.
    - List of treatment suggestions.
- **Rendering**: The frontend JavaScript dynamically updates the UI to show the results, images, and treatment advice without reloading the page.
