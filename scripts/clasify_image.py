import argparse
import pickle

from neural_network.utils import *
from settings import IMAGE_BINARY_FILE_NAME
from neural_network.utils import write_and_print

parser = argparse.ArgumentParser()
parser.add_argument('model_name', metavar='model_name')
parser.add_argument('image_path', metavar='image_path')


def classify_image(save_path, path_to_image):
    prepare_image(path_to_image)
    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    X = extract_data(f'dataset/{IMAGE_BINARY_FILE_NAME}-images-idx3-ubyte.gz', 1, 28)

    # normalization
    X -= int(np.mean(X))
    X /= int(np.std(X))
    X = X.reshape(1, 1, 28, 28)

    pred, prob = predict(X[0], f1, f2, w3, w4, b1, b2, b3, b4)
    write_and_print("---")
    write_and_print(f"Wynik klasyfikacji: {pred}")
    write_and_print(f"Prawdopodbienstwo: {prob}")
    write_and_print("---")


if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name
    image_path = args.image_path

    classify_image(model_name, image_path)
