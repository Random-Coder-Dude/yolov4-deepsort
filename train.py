import cv2
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models

# Set parameters
IMAGE_SIZE = (128, 128)  # Resize images for training
DATA_DIR = 'data'
BATCH_SIZE = 16
EPOCHS = 10
images = []
bboxes = []
drawing = False  # True when mouse is pressed
start_point = None  # Starting point for bounding box
end_point = None  # Ending point for bounding box

# Create a directory called 'data' if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Function to capture frames from the camera and annotate
def capture_and_annotate():
    global images, bboxes, drawing, start_point, end_point
    drawing = False  # True when mouse is pressed
    start_point = None  # Starting point for bounding box
    end_point = None  # Ending point for bounding box

    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    def draw_rectangle(event, x, y, flags, param):
        global start_point, drawing, end_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)  # Update the end point for live rectangle drawing
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)  # Finalize the end point
            # Save the bounding box coordinates
            bboxes.append([start_point[0], start_point[1], end_point[0], end_point[1]])
            cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)  # Draw rectangle permanently
            cv2.imshow("Frame", frame)

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # If drawing, keep the rectangle on the screen
        if drawing and start_point is not None and end_point is not None:
            img_copy = frame.copy()  # Make a copy of the current frame
            cv2.rectangle(img_copy, start_point, end_point, (255, 0, 0), 2)  # Draw the rectangle on the copy
            cv2.imshow("Frame", img_copy)  # Show the copy with the rectangle
        else:
            cv2.imshow("Frame", frame)  # Show the original frame

        # Capture the image when 'c' is pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Save the current frame as an image file in the 'data' folder
            img_resized = cv2.resize(frame, IMAGE_SIZE)
            images.append(img_resized)
            print("Image captured and annotated!")
            
            # Save the current frame as an image file in the 'data' folder
            filename = f"{DATA_DIR}/captured_image_{len(images)}.png"
            cv2.imwrite(filename, frame)  # Save the frame with bounding box
            print(f"Saved frame as '{filename}'")
            
            # Save bounding box coordinates to a corresponding text file
            bbox_filename = f"{DATA_DIR}/captured_image_{len(images)}.txt"
            with open(bbox_filename, 'w') as f:
                f.write(','.join(map(str, bboxes[-1])))  # Save the last drawn bounding box

        # Stop capturing when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load data for training
def load_data():
    images = []
    bboxes = []
    for img_path in glob.glob(os.path.join(DATA_DIR, '*.png')):
        # Load image
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize to [0, 1]
        images.append(image)
        
        # Load corresponding bounding box coordinates
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        bbox_path = os.path.join(DATA_DIR, f"{base_name}.txt")
        with open(bbox_path, 'r') as f:
            bbox = f.read().strip().split(',')
            bboxes.append([float(coord) for coord in bbox])  # Convert to float

    return np.array(images), np.array(bboxes)

# Build a simple model
def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4)  # Output layer for bounding box coordinates
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')  # Using MSE for bounding box regression
    return model

# Training function
def train_model():
    images, bboxes = load_data()
    model = build_model()
    model.fit(images, bboxes, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the trained model
    model.save('object_detection_model.keras')
    print("Model trained and saved as 'object_detection_model.keras'.")

# Main function to execute capturing and training
if __name__ == "__main__":
    capture_and_annotate()
    train_model()
