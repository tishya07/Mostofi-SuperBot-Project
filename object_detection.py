import cv2
import depthai as dai
import numpy as np
import os

# COCO labels
labelMap = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Blob file path 
BLOB_PATH = "yolov8n_coco_640x352_openvino_2022_1_6shave.blob"

# Verify blob file exists
if not os.path.exists(BLOB_PATH):
    print(f"ERROR: Blob file not found at {BLOB_PATH}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    exit(1)

print(f"Using blob: {BLOB_PATH}")

# Create pipeline
pipeline = dai.Pipeline()

# RGB Camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 352)  # Match blob input size
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

# Mono cameras for depth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(30)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(30)

# Stereo depth
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Neural network
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
spatialDetectionNetwork.setBlobPath(BLOB_PATH)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# YOLOv8 specific settings
spatialDetectionNetwork.setNumClasses(80)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
camRgb.preview.link(spatialDetectionNetwork.input)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

# Outputs
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

spatialDetectionNetwork.passthrough.link(xoutRgb.input)
spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Connect to device
print("Connecting to OakD...")
with dai.Device(pipeline) as device:
    print("Pipeline started successfully!")
    print("Detection running on-device at 30 FPS")
    print("Press 'q' to quit")
    
    # Output queues
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    startTime = None
    counter = 0
    fps = 0

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()
        detections = inDet.detections
        
        # Resize RGB frame to match depth frame size
        if depthFrame is not None and frame is not None:
            depth_height, depth_width = depthFrame.shape[:2]
            frame = cv2.resize(frame, (depth_width, depth_height))
        
        # Calculate FPS
        counter += 1
        if startTime is None:
            startTime = inPreview.getTimestamp()
        else:
            currentTime = inPreview.getTimestamp()
            fps = counter / (currentTime - startTime).total_seconds()

        # Draw detections
        for detection in detections:
            bbox = (
                int(detection.xmin * frame.shape[1]),
                int(detection.ymin * frame.shape[0]),
                int(detection.xmax * frame.shape[1]),
                int(detection.ymax * frame.shape[0])
            )
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            label = labelMap[detection.label] if detection.label < len(labelMap) else str(detection.label)
            
            # Get distance in centimeters
            x = detection.spatialCoordinates.x
            y = detection.spatialCoordinates.y
            z = detection.spatialCoordinates.z
            
            text = f"{label} {int(z/10)}cm"
            cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display FPS
        # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
        #          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show number of detections
        #cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Detection", frame)
        
        # Show depth with same size
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        cv2.imshow("Depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("\nSession ended")
