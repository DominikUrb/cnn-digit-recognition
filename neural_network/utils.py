import gzip
import os
import time
import urllib.request
from urllib.error import HTTPError
from array import *

from PIL import Image

from neural_network.forward import *
from settings import IMAGE_BINARY_FILE_NAME


def extract_data(filename, num_images, IMAGE_WIDTH):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH * IMAGE_WIDTH)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def initialize_filter(size, scale=1.0):
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weight(size):
    return np.random.standard_normal(size=size) * 0.01


def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs


def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s=1, pool_f=2, pool_s=2):
    conv1 = forward_convolution(image, f1, b1, conv_s)
    conv1[conv1 <= 0] = 0  # relu

    conv2 = forward_convolution(conv1, f2, b2, conv_s)
    conv2[conv2 <= 0] = 0  # relu

    pooled = max_pool(conv2, pool_f, pool_s)
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))

    z = w3.dot(fc) + b3  # first dense layer
    z[z <= 0] = 0  # relu

    out = w4.dot(z) + b4  # second dense layer
    probs = softmax(out)  # predict with softmax

    return np.argmax(probs), np.max(probs)


def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28, 28, 1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1, 28, 28, 1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data


def prepare_image(path_to_image):
    data_image = array('B')
    Im = Image.open(path_to_image)
    pixel = Im.load()
    width, height = Im.size

    for x in range(0, width):
        for y in range(0, height):
            data_image.append(pixel[y, x])

    # number of files in HEX
    hexval = "{0:#0{1}x}".format(1, 6)

    header = array('B')
    header.extend([0, 0, 8, 1, 0, 0])
    header.append(int('0x' + hexval[2:][:2], 16))
    header.append(int('0x' + hexval[2:][2:], 16))

    # additional header for images array
    if max([width, height]) <= 256:
        header.extend([0, 0, 0, width, 0, 0, 0, height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels');

    # changing MSB for image data (0x00000803)
    header[3] = 3

    data_image = header + data_image

    output_file = open(f'dataset/{IMAGE_BINARY_FILE_NAME}-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()

    os.system(f'rm dataset/{IMAGE_BINARY_FILE_NAME}-images-idx3-ubyte.gz')
    os.system(f'gzip dataset/{IMAGE_BINARY_FILE_NAME}-images-idx3-ubyte')


def write_and_print(text):
    f = open("out.txt", "a")
    f.write(text)
    print(text)


def write(text):
    f = open("out.txt", "a")
    f.write(text)


def download_test_datasets():
    while not os.path.isfile("dataset/t10k-images-idx3-ubyte.gz"):
        write_and_print("Rozpoczeto pobieranie t10k-images-idx3-ubyte.gz")
        try:
            urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                                       "dataset/t10k-images-idx3-ubyte.gz")
            write_and_print("Pobrano t10k-images-idx3-ubyte.gz")
        except HTTPError as e:
            write_and_print(f"Pobieranie nieudane, serwer zwrócił błąd: {e}")
            time.sleep(1)

    while not os.path.isfile("dataset/t10k-labels-idx1-ubyte.gz"):
        write_and_print("Rozpoczęto pobieranie t10k-labels-idx1-ubyte.gz")
        try:
            urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                                       "dataset/t10k-labels-idx1-ubyte.gz")
            write_and_print("Pobrano t10k-labels-idx1-ubyte.gz")
        except HTTPError as e:
            write_and_print(f"Pobieranie nieudane, serwer zwrócił błąd: {e}")
            time.sleep(1)


def download_train_datasets():
    while not os.path.isfile("dataset/train-images-idx3-ubyte.gz"):
        write_and_print("Rozpoczęto pobieranie train-images-idx3-ubyte.gz")
        try:
            urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                                   "dataset/train-images-idx3-ubyte.gz")
            write_and_print("Pobrano train-images-idx3-ubyte.gz")
        except HTTPError as e:
            write_and_print(f"Pobieranie nieudane, serwer zwrócił błąd: {e}")
            time.sleep(1)

    while not os.path.isfile("dataset/train-labels-idx1-ubyte.gz"):
        write_and_print("Rozpoczęto pobieranie train-labels-idx1-ubyte.gz")
        try:
            urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                                   "dataset/train-labels-idx1-ubyte.gz")
            write_and_print("Pobrano train-labels-idx1-ubyte.gz")
        except HTTPError as e:
            write_and_print(f"Pobieranie nieudane, serwer zwrócił błąd: {e}")
            time.sleep(1)
