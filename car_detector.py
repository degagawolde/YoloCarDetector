import cv2
from ultralytics import YOLO
import yt_dlp

model = YOLO("yolov8n.pt")

youtube_url = "https://youtu.be/MNn9qKG2UFI?si=Pt6RE8dt17OV67ne"

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(
                img,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                img,
                f"{result.names[int(box.cls[0])]}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1,
            )
    return img, results


def get_video_url(youtube_url):
    ydl_opts = {
        "format": "best",
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get("url", None)
        return video_url


video_url = get_video_url(youtube_url)
# Open video capture with OpenCV
cap = cv2.VideoCapture(video_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame as required (e.g., passing to YOLO)
    result_img, _ = predict_and_detect(model, frame, classes=[], conf=0.5)
    cv2.imshow("Video", result_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()