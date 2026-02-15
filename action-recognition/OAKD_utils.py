import cv2
import depthai as dai

# Create pipeline
def OAKDCamera_init():
	device = dai.Device()
	pipeline = dai.Pipeline(device)
	outputQueues = {}
	sockets = device.getConnectedCameras()
	for socket in sockets:
		cam = pipeline.create(dai.node.Camera).build(socket)
		outputQueues[str(socket)] = cam.requestFullResolutionOutput().createOutputQueue()

	pipeline.start()
	return pipeline, outputQueues

def fetch_video(pipeline, outputQueues):
	if not pipeline.isRunning():
		print("pipeline not running")
		
	while pipeline.isRunning():
		for name in outputQueues.keys():
			queue = outputQueues[name]
			videoIn = queue.get()
			assert isinstance(videoIn, dai.ImgFrame)
			# Visualizing the frame on slower hosts might have overhead
			cv2.imshow(name, videoIn.getCvFrame())

		if cv2.waitKey(1) == ord("q"):
			break
