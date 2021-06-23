from neural_network.network import *
from neural_network.utils import *

from tqdm import tqdm
import argparse
# import matplotlib.pyplot as plt
import pickle
from neural_network.utils import write_and_print

parser = argparse.ArgumentParser()
parser.add_argument('model_name', metavar='model_name')
parser.add_argument('test_data_q', metavar='test_data_q')
parser.add_argument('train_data_q', metavar='train_data_q')


def train_model(save_path, test_data_q=10000, train_data_q=50000, num_epochs=2):
    download_train_datasets()
    download_test_datasets()

    cost = train(save_path=save_path, train_data_q=train_data_q, num_epochs=num_epochs)

    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    write_and_print("----------------------")
    write_and_print("Przebieg funkcji strat")
    for c in cost:
        write_and_print(c)
    # Plot cost
    # plt.plot(cost, 'r')
    # plt.xlabel('# Iterations')
    # plt.ylabel('Cost')
    # plt.legend('Loss', loc='upper right')
    # plt.savefig("cost_loss.jpg")
    # plt.show()

    X = extract_data('dataset/t10k-images-idx3-ubyte.gz', test_data_q, 28)
    y_dash = extract_labels('dataset/t10k-labels-idx1-ubyte.gz', test_data_q).reshape(test_data_q, 1)

    # normalization
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
    write_and_print("----------------------")
    write_and_print("srednia dokladnosc: %.2f%%" % (float(corr / len(test_data) * 100)))
    x = np.arange(10)
    digit_recall = [x / y for x, y in zip(digit_correct, digit_count)]

    write_and_print("----------------------")
    write_and_print("Rozklad wartosci predykcji")
    write_and_print("[wynik klasyfikacji]     [% prawidlowych klasyfikacji]")
    for i in range(0, 10):
        write_and_print(f"{i}                         {digit_recall[i] * 100}")

    # plt.xlabel('Digits')
    # plt.ylabel('Recall')
    # plt.title("Recall on Test Set")
    # plt.bar(x, digit_recall)
    # plt.savefig("recall.jpg")
    # plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name
    test_data_q = args.test_data_q
    train_data_q = args.train_data_q

    train_model(model_name, test_data_q, train_data_q)
