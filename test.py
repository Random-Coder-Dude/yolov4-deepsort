import cv2
import numpy as np
import tensorflow as tf

# Parameters
IMAGE_SIZE = (128, 128)  # Resize images for testing
MODEL_PATH = 'object_detection_model.keras'

# Load the trained model
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Draw bounding box on the image
def draw_bounding_box(image, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# Test the model in real-time
def test_model():
    model = load_model()
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for prediction
        image_resized = cv2.resize(frame, IMAGE_SIZE) / 255.0  # Normalize to [0, 1]
        image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension

        # Make prediction
        predicted_bbox = model.predict(image_resized)[0]

        # Draw the predicted bounding box
        draw_bounding_box(frame, predicted_bbox)

        # Display the result
        cv2.imshow("Real-Time Object Detection", frame)

        # Stop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model()
