import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (use the appropriate model for better accuracy if needed)
model = YOLO('../yolov8n.pt')

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Perform inference on the current frame
    results = model.predict(source=frame)

    # Get the annotated frame (with bounding boxes and labels)
    annotated_frame = results[0].plot()

    # Count the number of people detected in the frame
    num_persons = sum(1 for obj in results[0].boxes.data if obj[-1] == 0)  # Class 0 is 'person' in YOLO

    # Display the number of persons detected on the frame
    cv2.putText(annotated_frame, f'Persons Detected: {num_persons}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with YOLOv8 detection
    cv2.imshow('YOLOv8 Detection - Webcam', annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
