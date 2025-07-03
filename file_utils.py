#################### GPT-4o mini CODE ####################
import struct
import numpy as np

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number and number of images
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels
#################### END ####################