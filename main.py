# main.py
import cv2
import argparse
import numpy
from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def plot_boxes(results, frame):
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x_min, y_min, x_max, y_max = map(int, box)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display label and confidence
            label = f"{int(class_id)}: {confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("finalsbest.pt")

    while True:
        ret, frame = cap.read()
        results = model(frame, show=True)

        frame = plot_boxes(results, frame)
        cv2.imshow("YOLOv8 Live", frame)
        if cv2.waitKey(30) == 27:
            break

if __name__ == "__main__":
    main()
