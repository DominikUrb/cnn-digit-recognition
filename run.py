from os import listdir
from os.path import isfile, join

from scripts.clasify_image import classify_image
from neural_network.utils import write_and_print
from scripts.test_model import measure_performance
from settings import MODELS_PATH
from scripts.train_model import train_model

go_to_main_menu_message = '\nAby powrocic do glownego menu wprowadz "q"'


def main():
    action = main_view_input()
    if action == "1":
        measure_performance_input()
    elif action == "2":
        classify_image_input()
    elif action == "3":
        train_model_input()
    elif action == "q":
        quit()


def main_view_input():
    control_message = """------------------------
W celu nawigacji wprowadz odpowiednia cyfre.
    1 - Pomiar dokladnosci dokonywanych predykcji modelu.
    2 - Klasyfikacja znaku na wskazanym obrazie.
    3 - Wytrenowanie nowego modelu CNN.
    q - Zakoncz dzialanie programu.
    """
    write_and_print(control_message)
    user_input = input()
    write_and_print(user_input)
    while user_input not in ("1", "2", "3", "q"):
        user_input = input()
        write_and_print(user_input)

    return user_input


def measure_performance_input():
    trained_models = get_trained_models()
    model_name_message = f"""------------------------
Aby dokonac pomiaru dokladnosci wytrenowanego modelu podaj nazwe wybranego modelu sposrod dostepnych:
    {get_trained_models_str(trained_models)}
{go_to_main_menu_message}
    """
    write_and_print(model_name_message)
    model_name = input()
    write_and_print(model_name)

    if not model_name.endswith('.pkl') and model_name != 'q' and model_name != '':
        model_name += '.pkl'

    if model_name == 'q':
        return main()
    elif model_name not in trained_models:
        measure_performance_input()
    else:
        amount_of_test_data_message = f"""------------------------
Wskaz wielkosc zbioru testowego na ktorym chcesz dokonac pomiaru w przedziale 1 - 10000. 
{go_to_main_menu_message}
        """
        amount_of_test_data = get_int_input(amount_of_test_data_message, min_value=1, max_value=10000)
        if amount_of_test_data == 'q':
            return main()
        else:
            measure_performance(f"models_parameters/{model_name}", amount_of_test_data)
            return main()


def classify_image_input():
    trained_models = get_trained_models()

    model_name_message = f"""------------------------
Aby dokonac klasyfikacji obrazu podaj nazwe wybranego modelu sposrod dostepnych:
    {get_trained_models_str(trained_models)}
{go_to_main_menu_message}
    """
    write_and_print(model_name_message)
    model_name = input()
    write_and_print(model_name)

    if not model_name.endswith('.pkl') and model_name != 'q':
        model_name += '.pkl'

    if model_name == 'q':
        return main()
    elif model_name not in trained_models:
        classify_image_input()
    else:
        found_image_by_path = False
        path_to_image_message = f"""------------------------
Podaj sciezke do obrazu w formacie '.jpg' ze znakiem, ktory chcesz poddac klasyfiakcji.
{go_to_main_menu_message}
        """
        while found_image_by_path is False:
            try:
                write_and_print(path_to_image_message)
                path_to_image = input()
                write_and_print(path_to_image)
                if path_to_image == 'q':
                    return main()
                if not isfile(path_to_image) or not path_to_image.endswith('.jpg'):
                    write_and_print("Wskazany adres jest nieprawidłowy")
                    continue
                classify_image(f"models_parameters/{model_name}", path_to_image)
                found_image_by_path = True
                return main()
            except:
                write_and_print("Wskazany adres jest nieprawidłowy")


def train_model_input():
    trained_models = get_trained_models()

    model_name_message = f"""------------------------
Aby wytrenowac nowy model oparty o siec CNN wprowadz jego nazwe.
{go_to_main_menu_message}
    """
    write_and_print(model_name_message)
    model_name = input()
    write_and_print(model_name)

    if not model_name.endswith('.pkl') and model_name != 'q':
        model_name += '.pkl'

    if model_name == 'q':
        return main()
    elif model_name in trained_models:
        write_and_print("Model o tej nazwie juz istnieje, podaj inna nazwe.")
        train_model_input()
    else:
        train_data_q_message = f"""------------------------
Podaj wielkosc zbioru treningowego w zakresie od 1 do 50000.
{go_to_main_menu_message}
        """

        test_data_q_message = f"""------------------------
Podaj wielkosc zbioru testowego w zakresie od 1 do 10000.
{go_to_main_menu_message}
        """

        num_epochs_message = f"""------------------------
Podaj ilosc iteracji uczenia (epochs).
{go_to_main_menu_message}
        """
        train_data_q = get_int_input(train_data_q_message, min_value=1, max_value=50000)
        if train_data_q == 'q':
            return main()
        test_data_q = get_int_input(test_data_q_message, min_value=1, max_value=10000)
        if test_data_q == 'q':
            return main()
        num_epochs = get_int_input(num_epochs_message, min_value=1)
        if num_epochs == 'q':
            return main()

        train_model(f"models_parameters/{model_name}", train_data_q, test_data_q, num_epochs)
        return main()


def get_int_input(message, min_value=None, max_value=None):
    value_int = None
    while value_int is None:
        try:
            write_and_print(message)
            v = input()
            write_and_print(v)
            if v == 'q':
                return 'q'

            v_int = int(v)

            if min_value is not None:
                if min_value > v_int:
                    continue
            if max_value is not None:
                if max_value < v_int:
                    continue

            value_int = v_int
        except:
            pass

    return value_int


def get_trained_models():
    return [f for f in listdir(MODELS_PATH) if isfile(join(MODELS_PATH, f))]


def get_trained_models_str(trained_models):
    trained_models_str = ""
    for trained_model in trained_models:
        trained_models_str += f"\n    - {trained_model}"
    return trained_models_str


if __name__ == '__main__':
    welcome_message = """------------------------
Niniejszy projekt umozliwia wytrenowanie modelu sztucznej sieci neuronowej, ktora umozliwi rozpoznawanie znakow pisanych - cyfr.
Dostepny jest juz wytrenowany model, ktory umozliwi natychmiastowe dokonanie predykcji z dokladnoscia okolo 98%.
    """
    write_and_print(welcome_message)
    main()
