from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
from flask_mysqldb import MySQL
import json
import time

app = Flask(__name__)
model = YOLO('finalsbest.pt')
cap = cv2.VideoCapture(0)  # Initialize cap here

app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "detection"

mysql = MySQL(app)

# Initialize object counts
unique_objects = set()
# Define the classes you want to detect
target_classes = {'person', 'door', 'staircase'}
elapsed_time = 0  # Initialize elapsed_time


def filter_results(results, frame):
    filtered_results = []

    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        filtered_boxes = []

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            label = model.names[int(class_id)]

            if label:
                # Check if the object is unique based on its bounding box
                box_key = tuple(box)
                if box_key not in unique_objects:
                    unique_objects.add(box_key)

                    # Update the count in the database
                    insert_detection_count("total", len(unique_objects))

                    # Draw bounding box and label
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                    label_text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Update the count in the database
                insert_detection_count(label, 1)

                filtered_boxes.append((box, confidence, class_id))

        filtered_results.append(filtered_boxes)

    return filtered_results


def detect_objects(frame):
    global unique_objects
    results = model.track(frame, persist=True)
    detections = filter_results(results, frame)

    for filtered_boxes in detections:
        for box, confidence, class_id in filtered_boxes:
            label = model.names[int(class_id)]
            if label in target_classes:
                new_object = True

                # Check if the bounding box is already in the set
                if tuple(box) in unique_objects:
                    new_object = False

                if new_object:
                    unique_objects.add(tuple(box))

                    # Update the count in the database
                    insert_detection_count("total", len(unique_objects))

                    # Draw bounding box and label
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label_text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Update the count in the database for the specific object type
                    insert_detection_count(label, 1)

    # Update the count in the database for the "total" object type
    insert_detection_count("total", sum(len(boxes) for boxes in detections))

    # Display real-time detection information on the frame
    draw_detection_info(frame)

    return frame, results


def draw_detection_info(frame):
    # Display specific object counts and elapsed time on the frame
    bounding_box_count = sum(len(boxes) for boxes in filter_results(model.track(frame), frame))
    insert_detection_count("total", bounding_box_count)

    info_text = f"Total Bounding Boxes: {bounding_box_count}, Elapsed Time: {elapsed_time:.1f}ms"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


@app.route('/')
def index():
    return render_template('index.html')  # You can add zones and classes if needed


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ... (previous code)


def generate_frames():
    global elapsed_time
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            start_time = time.time()
            frame, results = detect_objects(frame)
            elapsed_time = (time.time() - start_time) * 1000  # time in milliseconds

            # Convert PyTorch Tensor objects to NumPy arrays for serialization
            filtered_results = filter_results(results, frame)
            filtered_results_np = [
                [
                    (
                        box.tolist(),  # Convert NumPy array to Python list
                        float(confidence),
                        int(class_id)
                    )
                    for box, confidence, class_id in boxes
                ]
                for boxes in filtered_results
            ]

            detection_data = {
                "total_object_count": len(unique_objects),
                "elapsed_time": elapsed_time,
                "filtered_results": filtered_results_np
            }

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                b'Content-Type: application/json\r\n\r\n' + json.dumps(detection_data).encode() + b'\r\n'
            )

# ... (remaining code)


def overlaps_significantly(box1, box2, threshold=0.5):
    # Implement logic to check if box1 and box2 overlap significantly
    # For simplicity, you can use the Intersection over Union (IoU) method or any other method you prefer
    pass


def insert_detection_count(object_type, count):
    with app.app_context():
        cursor = mysql.connection.cursor()
        cursor.execute(
            "INSERT INTO detection_data (object_type, count) VALUES (%s, %s) ON DUPLICATE KEY UPDATE count = %s",
            (object_type, count, count)
        )
        mysql.connection.commit()
        cursor.close()


if __name__ == '__main__':
    with app.app_context():
        # Create the detection_data table if it doesn't exist
        with mysql.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    object_type VARCHAR(255) NOT NULL,
                    count INT DEFAULT 0,
                    UNIQUE KEY (object_type)
                )
                """
            )
            mysql.connection.commit()

    app.run(debug=True)
