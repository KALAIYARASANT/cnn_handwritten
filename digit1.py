import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

@st.cache_resource
def load_model():
    model = tf.saved_model.load(r'C:\Users\sures\OneDrive\Desktop\digit\my_model\my_model')
    return model.signatures["serving_default"]

infer = load_model()

st.title("🖌️ Multi-Digit Handwritten Recognition")

st.markdown("Draw digits (0-9) in the box and click **Predict** to classify all digits.")

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    cv2.imwrite("img.jpg", img)

def predict_multi_digits(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        st.error(f"Failed to load image from path: {image_path}")
        return []

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return []

    # Sort contours left-to-right for correct digit order
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours_sorted = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])]

    predictions = []
    input_key = list(infer.structured_input_signature[1].keys())[0]
    output_key = list(infer.structured_outputs.keys())[0]

    for cnt in contours_sorted:
        x, y, w, h = cv2.boundingRect(cnt)
        digit = th[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), mode='constant', constant_values=0)
        digit_input = padded_digit.reshape(1, 28, 28, 1).astype("float32") / 255.0

        pred = infer(**{input_key: tf.convert_to_tensor(digit_input)})[output_key].numpy()[0]
        predicted_digit = int(np.argmax(pred))
        predictions.append(predicted_digit)

    return predictions

if st.button("Predict"):
    digits = predict_multi_digits("img.jpg")
    if digits:
        st.success(f"🧠 Predicted digits (left to right): {' '.join(map(str, digits))}")
    else:
        st.warning("No digits detected. Please draw clearly.")
