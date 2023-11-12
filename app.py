from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load YOLO model
model = YOLO('finalsbest.pt')

# Open camera
cap = cv2.VideoCapture(0)

# Define the classes you want to detect
target_classes = {'person', 'door', 'stair'}

def filter_results(results):
    filtered_results = []

    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        filtered_boxes = []
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            label = model.names[int(class_id)]
            if label in target_classes:
                filtered_boxes.append((box, confidence, class_id))

        filtered_results.append(filtered_boxes)

    return filtered_results

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.track(frame, persist=True)
            filtered_results = filter_results(results)

            # Draw filtered results on the frame
            for filtered_boxes in filtered_results:
                for box, confidence, class_id in filtered_boxes:
                    x_min, y_min, x_max, y_max = map(int, box)

                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Display label and confidence
                    label = f"{model.names[int(class_id)]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
