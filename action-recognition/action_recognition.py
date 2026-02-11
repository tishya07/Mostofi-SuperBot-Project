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

class ActionRecognitionTensorRT:
    def __init__(self, engine_path, input_shape=(1, 3, 224, 224), sequence_length=16):
        """
        Initialize TensorRT engine for action recognition
        
        Args:
            engine_path: Path to the .engine file
            input_shape: Expected input shape (batch, channels, height, width)
            sequence_length: Number of frames for temporal sequence
        """
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Load TensorRT engine
        print("Initializing TensorRT runtime...")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"Loading engine from: {engine_path}")
        self.engine = self._load_engine()
        
        print("Creating execution context...")
        self.context = self.engine.create_execution_context()
        
        # Get binding information
        self.num_bindings = self.engine.num_io_tensors
        print(f"Number of bindings: {self.num_bindings}")
        
        # Allocate buffers
        print("Allocating buffers...")
        self.buffers = self._allocate_buffers()
        
        print("TensorRT engine initialized successfully!")
        
    def _load_engine(self):
        """Load TensorRT engine from file"""
        try:
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            engine = self.runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                raise RuntimeError("Failed to deserialize engine")
            return engine
        except Exception as e:
            raise RuntimeError(f"Failed to load engine: {e}")
    
    def _allocate_buffers(self):
        """Allocate host memory buffers for inputs and outputs"""
        buffers = []
        
        # TensorRT 10+ API
        for i in range(self.num_bindings):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                
            # Convert TensorRT dtype to numpy dtype
            dtype_map = {
                trt.DataType.FLOAT: np.float32,
                trt.DataType.HALF: np.float16,
                trt.DataType.INT8: np.int8,
                trt.DataType.INT32: np.int32,
                trt.DataType.BOOL: np.bool_
            }
            np_dtype = dtype_map.get(tensor_dtype, np.float32)
               
            # Calculate size
            size = trt.volume(tensor_shape)
                
            # Allocate host memory
            host_mem = np.empty(size, dtype=np_dtype)
                
            # Store buffer info
            buffer_info = {
                'name': tensor_name,
                'shape': tensor_shape,
                'dtype': np_dtype,
                'size': size,
                'host_memory': host_mem,
                'is_input': is_input
            }
                
            buffers.append(buffer_info)
                
            print(f"  Tensor {i}: {tensor_name}")
            print(f"    Shape: {tensor_shape}")
            print(f"    Type: {np_dtype}")
            print(f"    Is Input: {is_input}")
        
        return buffers
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for the model
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Preprocessed frame as numpy array
        """
        # Resize to model input size
        height, width = self.input_shape[2], self.input_shape[3]
        frame_resized = cv2.resize(frame, (width, height))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame_normalized = (frame_normalized - mean) / std
        
        # Transpose to CHW format (Channels, Height, Width)
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
        
        return frame_transposed
    
    def infer(self, frames):
        """
        Run inference on a sequence of frames
        
        Args:
            frames: List or array of preprocessed frames
            
        Returns:
            Model output predictions
        """
        # Prepare input data
        # Stack frames into sequence
        input_data = np.stack(frames, axis=0)
        
        # Add batch dimension if needed
        if len(input_data.shape) == 3:  # (frames, height, width)
            input_data = np.expand_dims(input_data, axis=0)
        
        # Ensure correct dtype
        input_buffer = self.buffers[0]
        input_data = input_data.astype(input_buffer['dtype'])
        
        # Flatten and copy to input buffer
        input_flat = input_data.ravel()
        np.copyto(input_buffer['host_memory'][:len(input_flat)], input_flat)
        
        # Create binding addresses
        # TensorRT needs device pointers, but we'll use the execute_v2 method
        # which works with host memory when using the Python API without CUDA
        binding_addrs = []
        for buffer in self.buffers:
            # Get the address of the numpy array's data
            addr = buffer['host_memory'].ctypes.data
            binding_addrs.append(addr)
        
        # Execute inference
        # Note: execute_v2 handles memory transfers internally
        success = self.context.execute_v2(bindings=binding_addrs)
        
        if not success:
            raise RuntimeError("Inference execution failed")
        
        # Get output
        output_buffer = self.buffers[1]  # Assuming output is binding 1
        output = output_buffer['host_memory'].copy()
        
        # Reshape output if needed
        output_shape = output_buffer['shape']
        if output_shape[0] == -1:  # Dynamic batch size
            output_shape = (1,) + tuple(output_shape[1:])
        
        try:
            output = output.reshape(output_shape)
        except:
            pass  # Keep flat if reshape fails
        
        return output
    
    def add_frame(self, frame):
        """Add frame to buffer for temporal sequence"""
        preprocessed = self.preprocess_frame(frame)
        self.frame_buffer.append(preprocessed)
        
    def is_ready(self):
        """Check if enough frames are buffered for inference"""
        return len(self.frame_buffer) == self.sequence_length
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine
        if hasattr(self, 'runtime'):
            del self.runtime


def make_oakd_rgb_queue(preview_size=(640, 480), fps=30):
    pipeline = dai.Pipeline()

    # ColorCamera still works, but deprecated (warning is OK)
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(*preview_size)
    cam.setInterleaved(False)
    cam.setFps(fps)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # DepthAI v3: no explicit XLinkOut. Create queue directly from output.
    q = cam.preview.createOutputQueue(maxSize=4, blocking=False)

    # Start pipeline (replaces dai.Device(pipeline))
    pipeline.start()

    return pipeline, q


def softmax(x):
    """Apply softmax to get probabilities"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def main():
    # Configuration
    ENGINE_PATH = "models/actionrecognitionnet.engine"  # Path to your TensorRT engine
    
    # Common action recognition labels (update based on your model)
    ACTION_LABELS = [
        "walking", "riding bike", "running", "falling on floor", "pushing"
    ]
    
    # Model configuration
    SEQUENCE_LENGTH = 16  # Number of frames for action recognition
    INPUT_SHAPE = (1, 3, 224, 224)  # (batch, channels, height, width)
    CAMERA_FPS = 30
    CAMERA_RESOLUTION = (1920, 1080)
    
    # Display configuration
    CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display prediction
    
    print("="*60)
    print("OAK-D Action Recognition with TensorRT (No PyCUDA)")
    print("="*60)
    
    # Initialize action recognition inference
    try:
        action_recognizer = ActionRecognitionTensorRT(
            engine_path=ENGINE_PATH,
            input_shape=INPUT_SHAPE,
            sequence_length=SEQUENCE_LENGTH
        )
    except Exception as e:
        print(f"\n❌ Error loading TensorRT engine: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the engine file exists at:", ENGINE_PATH)
        print("2. Verify TensorRT is properly installed")
        print("3. Check that the engine was built for your TensorRT version")
        print("4. Ensure you have an NVIDIA GPU with CUDA support")
        return
    
    # Initialize OAK-D camera
    print("\n" + "="*60)
    print("Initializing OAK-D camera...")
    try:
        pipeline, q_rgb = make_oakd_rgb_queue();
    except Exception as e:
        print(f"\n❌ Error initializing camera: {e}")
        print("\nTroubleshooting:")
        print("1. Check if OAK-D camera is connected")
        print("2. Try a different USB port")
        print("3. Run with sudo (Linux) if permission denied")
        return
    
    # Start pipeline
    try:
        with dai.Device(pipeline) as device:
            print("✓ Camera started successfully!")
            print("\nControls:")
            print("  - Press 'q' to quit")
            print("  - Press 's' to save current frame")
            print("\n" + "="*60)
            
            # Get output queue
            #q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            # Statistics
            frame_count = 0
            inference_count = 0
            start_time = time.time()
            inference_times = []
            
            # For saving frames
            save_counter = 0
            
            # Current prediction (for smoothing)
            current_action = "Waiting..."
            current_confidence = 0.0
            
            while True:
                # Get frame from camera
                in_rgb = q_rgb.get()
                if in_rgb is None:
                    continue
                    
                frame = in_rgb.getCvFrame()
                if frame is None:
                    continue
                
                # Add frame to buffer
                action_recognizer.add_frame(frame)
                frame_count += 1
                
                # Run inference when buffer is full
                if action_recognizer.is_ready():
                    try:
                        # Measure inference time
                        inf_start = time.time()
                        output = action_recognizer.infer(list(action_recognizer.frame_buffer))
                        inf_time = time.time() - inf_start
                        inference_times.append(inf_time)
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
                        if confidence >= CONFIDENCE_THRESHOLD:
                            current_action = ACTION_LABELS[predicted_class] if predicted_class < len(ACTION_LABELS) else "unknown"
                            current_confidence = confidence
                        
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
                
                # Draw confidence bar
                if current_confidence > 0:
                    conf_text = f"Confidence: {current_confidence:.1%}"
                    cv2.putText(frame, conf_text, (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Draw confidence bar
                    bar_width = int(400 * current_confidence)
                    cv2.rectangle(frame, (20, 95), (420, 110), (100, 100, 100), -1)
                    cv2.rectangle(frame, (20, 95), (20 + bar_width, 110), (0, 255, 0), -1)
                
                # Draw statistics
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 200, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Inference time
                if inference_times:
                    avg_inf = np.mean(inference_times[-10:]) * 1000  # Last 10 inferences
                    cv2.putText(frame, f"Inference: {avg_inf:.1f}ms", 
                               (frame.shape[1] - 250, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Frame counter
                buffer_status = f"Buffer: {len(action_recognizer.frame_buffer)}/{SEQUENCE_LENGTH}"
                cv2.putText(frame, buffer_status, (frame.shape[1] - 220, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                
                # Show frame
                cv2.imshow("OAK-D Action Recognition (TensorRT)", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_counter += 1
                    filename = f"action_frame_{save_counter:04d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")
            
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "="*60)
            print("Session Statistics")
            print("="*60)
            print(f"Total frames processed: {frame_count}")
            print(f"Total inferences: {inference_count}")
            print(f"Average FPS: {fps:.2f}")
            if inference_times:
                avg_inf_time = np.mean(inference_times) * 1000
                min_inf_time = np.min(inference_times) * 1000
                max_inf_time = np.max(inference_times) * 1000
                print(f"Average inference time: {avg_inf_time:.2f}ms")
                print(f"Min inference time: {min_inf_time:.2f}ms")
                print(f"Max inference time: {max_inf_time:.2f}ms")
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure OAK-D camera is properly connected")
        print("2. Check USB connection and try a different port")
        print("3. Verify camera permissions (try running with sudo on Linux)")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
