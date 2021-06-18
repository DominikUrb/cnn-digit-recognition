from neural_network.utils import *

from tqdm import tqdm
import argparse
import pickle

from neural_network.utils import write_and_print


parser = argparse.ArgumentParser()
parser.add_argument('model_name', metavar='model_name')


def measure_performance(save_path, m=10000):
    download_test_datasets()

    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    X = extract_data('dataset/t10k-images-idx3-ubyte.gz', m, 28)
    y_dash = extract_labels('dataset/t10k-labels-idx1-ubyte.gz', m).reshape(m, 1)

    X -= int(np.mean(X))
    X /= int(np.std(X))
    test_data = np.hstack((X, y_dash))

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:, -1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    write_and_print("Mierzenie dokladnosci modelu na zbiorze testowym:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])] += 1
        if pred == y[i]:
            corr += 1
            digit_correct[pred] += 1

        t.set_description("Dokladnosc: %0.2f%%" % (float(corr / (i + 1)) * 100))

    write_and_print("Srednia dokladnosc: %.2f%%" % (float(corr / len(test_data) * 100)))


if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name
    measure_performance(model_name)
