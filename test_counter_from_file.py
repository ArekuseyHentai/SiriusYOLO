from modules.ultralytics import YOLO
from modules.ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("./src/file5.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(
    "object_counting_output.avi",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h))
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS))

# Define region points
region_points = [
    (524, 0),
    (
        524,
        586
    )
    ]
classes_to_count = [0]

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Обнаружен пустой кадр.")
        break
    tracks = model.track(
        im0,
        persist=True,
        show=False,
        classes=classes_to_count,
        verbose=False
        )
    #print(len(tracks[0].boxes))
    cv2.putText(
        im0,
        str(len(tracks[0].boxes)),
        (524, 524),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
        cv2.LINE_AA)
    im0 = counter.start_counting(im0, tracks)
    
    #print(counter.out_counts, counter.in_counts)
    
    video_writer.write(im0)

cap.release()
cv2.destroyAllWindows()
# EOF
