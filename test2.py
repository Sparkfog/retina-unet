

import tensorflow as tf
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("CUDA Available:", tf.test.is_built_with_cuda())

