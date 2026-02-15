import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class ActionRecognitionTensorRT:
    """
    DeepStream-free TensorRT runner for NVIDIA TAO ActionRecognitionNet-style models.

    Expected input tensor shape:
        (N, C, T, H, W) = (1, 3, 32, 224, 224)

    Meaning:
        - C=3 RGB channels
        - T=32 frames in the clip (temporal dimension)
        - H=W=224 spatial resolution
    """

    def __init__(self, engine_path: str, sequence_length: int = 32):
        self.engine_path = engine_path
        self.sequence_length = sequence_length
        self.frame_buffer = np.zeros((3, sequence_length, 224, 224), dtype=np.float32) 
        self.curr_frame_num = 0
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()

        # Discover input/output tensor names (TensorRT 10+)
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1 or len(self.output_names) != 1:
            raise RuntimeError(
                f"Expected 1 input and 1 output tensor, got inputs={self.input_names}, outputs={self.output_names}"
            )

        self.in_name = self.input_names[0]
        self.out_name = self.output_names[0]

        # Set concrete input shape if the engine is dynamic
        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        if any(d < 0 for d in in_shape):
            in_shape = (1, 3, self.sequence_length, 224, 224)
            self.context.set_input_shape(self.in_name, in_shape)

        # Fetch resolved shapes from the context
        self.in_shape = tuple(self.context.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.context.get_tensor_shape(self.out_name))

        # Get data types
        in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))

        # Calculate total number of elements for allocation
        # in_shape should be (1, 3, 32, 224, 224) - full shape with batch
        input_size = trt.volume(self.in_shape)
        output_size = trt.volume(self.out_shape)

        # Allocate host (CPU) pinned memory buffers
        self.h_input = cuda.pagelocked_empty(input_size, dtype=in_dtype)
        self.h_output = cuda.pagelocked_empty(output_size, dtype=out_dtype)

        # Allocate device (GPU) memory buffers
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # Bind device addresses to context (TensorRT 10+)
        self.context.set_tensor_address(self.in_name, int(self.d_input))
        self.context.set_tensor_address(self.out_name, int(self.d_output))

        # Create CUDA stream for async operations
        self.stream = cuda.Stream()

        print("TensorRT initialized")
        print(f"  Input : {self.in_name}  shape={self.in_shape} dtype={in_dtype}")
        print(f"  Output: {self.out_name} shape={self.out_shape} dtype={out_dtype}")
        print(f"  Input buffer size: {input_size} elements ({self.h_input.nbytes} bytes)")
        print(f"  Output buffer size: {output_size} elements ({self.h_output.nbytes} bytes)")

    def _load_engine(self):
        """Load the serialized TensorRT engine from file."""
        with open(self.engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        return engine

    @staticmethod
    def preprocess_frame(frame_bgr: np.ndarray, size=(224, 224)) -> np.ndarray:
        """
        Convert a BGR uint8 OpenCV frame into a normalized CHW float32 RGB tensor.
        
        Args:
            frame_bgr: OpenCV BGR image (HWC format)
            size: Target size (width, height)
            
        Returns:
            Preprocessed frame with shape (3, 224, 224)
        """
        # Resize frame
        frame = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB and normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame = (frame - mean) / std

        # Convert from HWC to CHW
        return np.transpose(frame, (2, 0, 1))

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Add one camera frame to the rolling clip buffer.
        
        Args:
            frame_bgr: OpenCV BGR image to add to buffer
        """
        if self.curr_frame_num < self.sequence_length:    
            curr_frame = self.preprocess_frame(frame_bgr)
            self.frame_buffer[:, self.curr_frame_num, :, :] = curr_frame
            self.curr_frame_num += 1

    def is_ready(self) -> bool:
        """
        Check if we have buffered enough frames for inference.
        
        Returns:
            True when we have exactly sequence_length frames buffered
        """
        ready = self.curr_frame_num == self.sequence_length
        if ready:
            print(f"Buffer ready with {self.curr_frame_num} frames")
            print(self.frame_buffer.shape)
        return ready

    def infer(self) -> np.ndarray:
        """
        Run inference using the current buffered clip.

        Returns:
            Output array shaped like self.out_shape (typically class probabilities/logits)
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Not enough frames buffered. Have {self.curr_frame_num}, need {self.sequence_length}"
            )
        
        print("Running inference...")
        
        # Prepare input based on expected shape
        # frame_buffer shape: (3, 32, 224, 224)
        # self.in_shape might be (1, 3, 32, 224, 224) or (3, 32, 224, 224)
        
        if self.in_shape[0] == 1:
            # Need to add batch dimension
            input_array = np.expand_dims(self.frame_buffer, axis=0)
        else:
            # Batch dimension not needed or already implicit
            input_array = self.frame_buffer
        
        print(f"  Frame buffer shape: {self.frame_buffer.shape}")
        print(f"  Input array shape: {input_array.shape}")
        print(f"  Expected shape: {self.in_shape}")
        print(f"  Flattened size: {input_array.size}, h_input size: {self.h_input.size}")
        
        # Copy input data to pinned host memory
        np.copyto(self.h_input, input_array.ravel())
        
        # Transfer input data from host (CPU) to device (GPU) asynchronously
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Execute inference asynchronously on GPU
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Transfer output data from device (GPU) back to host (CPU) asynchronously
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Wait for all GPU operations to complete
        self.stream.synchronize()
        
        # Reshape output to proper shape (remove batch dimension if needed)
        output = self.h_output.reshape(self.out_shape)
        
        # Reset buffer for next clip
        self.reset_buffer()
        
        print(f"Inference complete. Output: {output}")
        return output

    def reset_buffer(self) -> None:
        """Reset the frame buffer and counter for the next clip."""
        self.frame_buffer = np.zeros((3, self.sequence_length, 224, 224), dtype=np.float32) 
        self.curr_frame_num = 0

    def __del__(self):
        """Cleanup GPU resources."""
        try:
            if hasattr(self, 'd_input'):
                self.d_input.free()
            if hasattr(self, 'd_output'):
                self.d_output.free()
        except:
            pass
            
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities from logits.
    
    Args:
        x: Input logits array
        
    Returns:
        Probability distribution (sums to 1)
    """
    x = x - np.max(x)  # Subtract max for numerical stability
    e = np.exp(x)
    return e / np.sum(e)
