from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
from ultralytics import solutions

#load a pretrainedc YOLOv8n model
model = YOLO('BubbleModel1.pt')

#Run inference on the source
#results = model(source='IMG_5222.mp4', show=True, conf=0.5, save=True) #generator of results

PATH = "BubbleVideo.mp4"
cap = cv2.VideoCapture(PATH)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(w-30, 0), (w-30, h)]  # For line counting
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # For rectangle region counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # For polygon region counting

# Video writeri
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="BubbleModel1.pt", 
    show_in=True,  # Display in counts
    #show_out=True,  # Display out counts
     line_width=2,  # Adjust the line width for bounding boxes and text display
)


region_speed = [(w-100, 0), (w-100, h)]  # For line counting
speed = solutions.SpeedEstimator(
    #show=True,  # Display the output
    model="BubbleModel1.pt",  # Path to the YOLO11 model file.
    region=region_speed,  # Pass region points
    # classes=[0, 2],  # If you want to estimate speed of specific classes.
    line_width=2,  # Adjust the line width for bounding boxes and text display
)


# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    
    # Track objects in the frame
    tracks = model.track(im0, persist=True, show=False)
    out = speed.estimate_speed(im0)
    video_writer.write(out)
    for track in tracks:
        print(f"Bubbles exceeding speed: {track.speed}")


cap.release()
video_writer.release()
cv2.destroyAllWindows()

