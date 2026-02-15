"""
OAK-D Camera Action Recognition with TensorRT (No PyCUDA)
Runs inference on video stream from OAK-D camera using pre-trained TensorRT model
Uses ctypes and numpy for memory management instead of PyCUDA
"""

#Currently this doesn't work. need to fix oakd camera setup
import cv2
import numpy as np
import depthai as dai
import tensorrt as trt
from collections import deque
import time
import ctypes
from TAO_model_utils import ActionRecognitionTensorRT, softmax
from OAKD_utils import OAKDCamera_init

def main():
	# Configuration
	ENGINE_PATH = "models/rgb_action_net.engine"  # Path to your TensorRT engine
    
	# Common action recognition labels (update based on your model)
	ACTION_LABELS = [
		"walking", "riding bike", "running", "falling on floor", "pushing"
	]
    
	# Model configuration
	SEQUENCE_LENGTH = 32  # Number of frames for action recognition
	INPUT_SHAPE = (1, 3, 224, 224)  # (batch, channels, height, width)
	CAMERA_FPS = 30
	CAMERA_RESOLUTION = (1920, 1080)
    
	# Display configuration
	CONFIDENCE_THRESHOLD = 0.95  # Minimum confidence to display prediction
    
	print("="*60)
	print("OAK-D Action Recognition with TensorRT")
	print("="*60)
    
	# Initialize action recognition inference
	action_recognizer = ActionRecognitionTensorRT(
		engine_path=ENGINE_PATH,
		sequence_length=SEQUENCE_LENGTH
	)
    
	# Initialize OAK-D camera
	print("\n" + "="*60)
	print("Initializing OAK-D camera...")
	pipeline, queues = OAKDCamera_init();
	for name in queues.keys():
		print(name)
	# Start pipeline

	frame_count = 0
	inference_count = 0
	start_time = time.time()
	inference_times = []
        
	# For saving frames
	save_counter = 0
         
	# Current prediction (for smoothing)
	current_action = "Waiting..."
	current_confidence = 0.0
            
	while pipeline.isRunning:
		# Get frame from camera
		in_rgb = queues["CameraBoardSocket.CAM_A"].get()
		if in_rgb is None:
			continue
                    
		frame = in_rgb.getCvFrame()
		if frame is None:
			continue
		            
		# Add frame to buffer
		action_recognizer.add_frame(frame)
		frame_count += 1
		cv2.imshow("video", frame)
		# Run inference when buffer is full
		if action_recognizer.is_ready():
			try:
				print("infering")
				output = action_recognizer.infer()
				inference_count += 1
                        
				# Flatten output if needed
				if len(output.shape) > 1:
					output = output.flatten()
                        
				# Apply softmax to get probabilities
				probs = softmax(output[:len(ACTION_LABELS)])
                        
				# Get predicted class
				predicted_class = np.argmax(probs)
				confidence = probs[predicted_class]
                        
				# Update current prediction if confidence is high enough
				current_action = "unknown"
				if confidence >= CONFIDENCE_THRESHOLD:
					current_action = ACTION_LABELS[predicted_class] if predicted_class < len(ACTION_LABELS) else "unknown"
				print("Current action: ", current_action)
				print("Current confidence: ", confidence)
                        
			except Exception as e:
				print(f"\n⚠ Inference error: {e}")
				current_action = "Error"
				current_confidence = 0.0
                
			# Draw results on frame
			# Create a semi-transparent overlay for text background
			overlay = frame.copy()
                
			# Draw main prediction box
			box_height = 120
			cv2.rectangle(overlay, (0, 0), (frame.shape[1], box_height), (0, 0, 0), -1)
			frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                
			# Draw action label
			action_text = f"Action: {current_action}"
			cv2.putText(frame, action_text, (20, 40), 
			cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
		
            
            
	cv2.destroyAllWindows()
            
	# Print final statistics
	print("\n" + "="*60)
	print("Session Statistics")
	print("="*60)
	print(f"Total frames processed: {frame_count}")
	print(f"Total inferences: {inference_count}")
	print(f"Average FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
