import numpy as np
import os
import math


def get_feature_label_arr(dataset):
    feature_arr = []
    label_arr = []
    for line in dataset:
        let = line.split(",")
        feature, label = gen_feature_label(let)
        feature = list(map(float, feature))
        feature_arr.append(feature)
        label_arr.append(label)
    return feature_arr, label_arr


def get_test_features_arr(dataset):
    feature_arr = []
    for line in dataset:
        let = line.split(",")
        let = list(map(float, let))
        feature_arr.append(let)
    return feature_arr


def gen_feature_label(line):
    return line[:-1], line[-1]


def divide_classes(feature_arr, label_arr):
    classes = {}
    for i in range(len(label_arr)):
        classes[label_arr[i]] = []
        classes[label_arr[i]].append(feature_arr)
    return classes


def mean(values):
    return sum([int(x) for x in values]) / float(len(values))


def std(values):
    avg = mean(values)
    variance = sum([pow(x - avg, 2) for x in values]) / float(len(values) - 1)
    if variance != 0:
        return math.sqrt(variance)
    else:
        return 1


def dataset_attributes(feature_arr):
    attributes = [(mean(values), std(values)) for values in zip(*feature_arr)]
    return attributes


def train_NB(feature_arr, label_arr):
    class_dict = divide_classes(feature_arr, label_arr)
    class_attr = {}
    for class_label, data in class_dict.items():
        class_attr[class_label] = dataset_attributes(data[0])
    return class_attr


def get_probability(val, mean, std):
    exp = math.exp(-(math.pow(val - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp


def attr_class_probability(input_vector, class_dict):
    probability_dict = {}
    for classValue, class_attr in class_dict.items():
        probability_dict[classValue] = 1
        for i in range(len(class_attr)):
            mean, std = class_attr[i]
            x = input_vector[i]
            probability_dict[classValue] *= get_probability(x, mean, std)
    return probability_dict


def classify_NB(class_dict, datapoint):
    probability_dict = attr_class_probability(datapoint, class_dict)
    predicted_label, prob = None, -1
    for classValue, probability in probability_dict.items():
        if predicted_label is None or probability > prob:
            prob = probability
            predicted_label = classValue
    return predicted_label


def make_prediction(class_dict, test_data):
    predictions = []
    for i in range(len(test_data)):
        result = classify_NB(class_dict, test_data[i])
        predictions.append(result)
    return predictions


def calc_accuracy(test_data_labels, predictions):
    c = 0
    for i in range(len(test_data_labels)):
        if test_data_labels[i] == predictions[i]:
            c += 1
    return c / float(len(test_data_labels)) * 100


def split_test_train_set(dataset, split_point):
    splitted = np.split(np.array(dataset), [split_point])
    test_data = splitted[0].tolist()
    train_data = splitted[1].tolist()
    return train_data, test_data


def ten_fold_cross_validation(dataset):
    len_dataset = len(dataset)
    splt = int(len_dataset / 10)
    accuracy_arr = []
    for i in range(10):
        train_data, test_data = split_test_train_set(dataset, splt)
        np.random.shuffle(dataset)
        train_data_features_arr, train_data_labels_arr = get_feature_label_arr(train_data)
        test_data_features_arr, test_data_labels_arr = get_feature_label_arr(test_data)
        trained_dict = train_NB(train_data_features_arr, train_data_labels_arr)
        prediction_list = make_prediction(trained_dict, test_data_features_arr)
        accuracy = calc_accuracy(test_data_labels_arr, prediction_list)
        accuracy_arr.append(accuracy)
    return accuracy_arr


def testing(dataset, test_data_features_arr):
    train_data_features_arr, train_data_labels_arr = get_feature_label_arr(dataset)
    trained_dict = train_NB(train_data_features_arr, train_data_labels_arr)
    prediction_list = make_prediction(trained_dict, test_data_features_arr)
    return prediction_list


if __name__ == '__main__':
    directory = os.path.dirname(__file__)
    rel_path_train = "/train.txt"
    rel_path_test = "/test.txt"
    abs_file_path_train = directory + rel_path_train
    abs_file_path_test = directory + rel_path_test
    dataset_train = []
    dataset_test = []

    # training dataset
    with open(abs_file_path_train, "r") as datafile:
        for line in datafile:
            dataset_train.append(line)

    for i in range(len(dataset_train)):
        dataset_train[i] = dataset_train[i].rstrip('\n')
    # testing dataset
    with open(abs_file_path_test, "r") as datafile:
        for line in datafile:
            dataset_test.append(line)

    for i in range(len(dataset_test)):
        dataset_test[i] = dataset_test[i].rstrip('\n')
    test_data_features_arr = get_test_features_arr(dataset_test)
    ten_fold_accuracy = ten_fold_cross_validation(dataset_train)
    avg_accuracy = np.average(ten_fold_accuracy)
    print(avg_accuracy)
    predictions = testing(dataset_train, test_data_features_arr)
    print(predictions)
    f = open('labels.txt', 'w')
    for i in predictions:
        print(i, file=f)
