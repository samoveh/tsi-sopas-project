import streamlit as st
import numpy as np
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="F2Qjw6ol597K3W4eu89s"  # Replace with your actual API key
)

# Function to make predictions using Roboflow API
def predict(image):
    # Send image to Roboflow API
    result = CLIENT.infer(image, model_id="sopas/10")  
    return result

# Streamlit app
st.title("YOLOv8 Object Detection with Roboflow and Streamlit")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Make prediction
    st.write("Predicting...")
    predictions = predict(image)

    # Display predictions
    st.write("Predictions:")
    st.json(predictions)

    # Convert image to OpenCV format (NumPy array)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Fix color format

    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # Draw bounding boxes and labels
    for prediction in predictions.get('predictions', []):
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        label, confidence = prediction['class'], prediction['confidence']

        # Convert center coordinates to top-left and bottom-right for OpenCV
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))

        # Draw bounding box
        cv2.rectangle(image_np, start_point, end_point, (0, 255, 0), 2)

        # Create label text with confidence score
        label_text = f"{label}: {confidence:.2f}"

        # Position the text at the bottom of the bounding box
        text_offset_x = start_point[0]
        text_offset_y = end_point[1] + 20  # Below the box

        # Get text size for background
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        box_coords = ((text_offset_x, text_offset_y - text_height - 5), (text_offset_x + text_width, text_offset_y + 5))

        # Draw background rectangle for text
        cv2.rectangle(image_np, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)

        # Put label text
        cv2.putText(image_np, label_text, (text_offset_x, text_offset_y), font, font_scale, (0, 0, 0), font_thickness)

    # Convert back to RGB before displaying in Streamlit
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    st.image(image_np, caption='Predicted Image with Labels.', use_column_width=True)
