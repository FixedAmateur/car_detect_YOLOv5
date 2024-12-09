import cv2
import torch

# Load the YOLOv5 model (small version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the video capture (you can change this to your video file path)
video_path = 'data/car_videos/car_1.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Camera focal length (in pixels) and known object height (real-world, in meters)
focal_length = 700  # This is an example value, adjust according to your camera's specification
object_real_height = 1.5  # Average car height in meters

# Set up the confidence threshold for detections
confidence_threshold = 0.25
warning_distance = 10

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform YOLOv5 inference on the current frame
    results = model(frame)

    # Initialize the warning message and object name
    warning_message = ''
    close_object_name = ''
    close_object_distance = 0

    # Apply the confidence threshold and draw bounding boxes
    for *box, conf, cls in results.xyxy[0]:  # xyxy coordinates, confidence, and class
        if conf >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate object height in pixels
            object_height_pixels = y2 - y1

            # Estimate distance using the formula
            distance = (focal_length * object_real_height) / object_height_pixels

            # Display the distance on the frame
            cv2.putText(frame, f'Distance: {distance:.2f}m', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

            # Check if the object is too close
            if distance < warning_distance:
                warning_message = 'WARNING: Too close!'
                close_object_name = model.names[int(cls)]  # Get the object name
                close_object_distance = distance
                cv2.putText(frame, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # If an object is too close, display its name and distance in the console
    if close_object_name:
        warning_in_console = f'{close_object_name} is too close! Distance: {close_object_distance:.2f} meters'
        print(warning_in_console)

        # Display its name in the frame
        cv2.putText(frame, f'{close_object_name} is too close!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2)

    # Show the frame with detections and distance info
    cv2.imshow('Real-Time Detection with Distance', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
