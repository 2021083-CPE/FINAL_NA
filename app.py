from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import mysql.connector
from flask_mysqldb import MySQL
import json

app = Flask(__name__)
model = YOLO('finalsbest.pt')
cap = cv2.VideoCapture(0)  # Initialize cap here

app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "detection"

mysql = MySQL(app)

# Define the classes you want to detect
target_classes = {'person', 'door', 'star'}  # Assuming 'child' is one of the target classes

# Define polygonal zones as a list of vertices (x, y)
polygon_zones = {
    'zone1': [(100, 200), (150, 250), (200, 200), (150, 150)],
    'zone2': [(300, 100), (350, 150), (400, 100), (350, 50)],
    # Add more zones as needed
}

# Initialize object counts
object_counts = {class_name: 0 for class_name in target_classes}
object_counts.update({f'{class_name}_{zone_name}': 0 for class_name in target_classes for zone_name in polygon_zones})


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

                # Increment object count for the detected class
                object_counts[label] += 1

                # Check if the object is within any polygonal zone
                for zone_name, zone_vertices in polygon_zones.items():
                    if is_point_in_polygon((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, zone_vertices):
                        # Increment object count for the zone
                        object_counts[f'{label}_{zone_name}'] += 1

        filtered_results.append(filtered_boxes)

    return filtered_results

def is_point_in_polygon(x, y, vertices):
    # Check if a point is inside a polygon using the ray-casting algorithm
    # This function can be implemented or you can use a library like Shapely
    pass


def detect_objects(frame):
    results = model.track(frame, persist=True)

    # Draw filtered results on the frame
    for filtered_boxes in filter_results(results):
        for box, confidence, class_id in filtered_boxes:
            x_min, y_min, x_max, y_max = map(int, box)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display label and confidence
            label = f"{model.names[int(class_id)]}: {confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_objects(frame)

            # Convert object_counts to JSON
            object_counts_json = json.dumps(object_counts)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                   b'Content-Type: application/json\r\n\r\n' + object_counts_json.encode() + b'\r\n')


def insert_detection(object_type, processing_time):
    with app.app_context():
        cursor = mysql.connection.cursor()
        cursor.execute(
            "INSERT INTO detection_data (object_type, processing_time) VALUES (%s, %s)",
            (object_type, processing_time)
        )
        mysql.connection.commit()
        cursor.close()


@app.route('/')
def index():
    return render_template('index.html', zones=polygon_zones, classes=target_classes)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    with app.app_context():
        # Create the detection_data table if it doesn't exist
        with mysql.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    object_type VARCHAR(255) NOT NULL,
                    processing_time FLOAT NOT NULL,
                    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
    app.run(debug=True)
