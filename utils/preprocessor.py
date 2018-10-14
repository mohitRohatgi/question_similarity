import os
import numpy as np
import pickle
from config.config import Config
config = Config()


# checkout test for usage.
def load_vocab():
    resources_dir = os.path.join(os.getcwd(), 'resources')
    char_to_int_path = os.path.join(resources_dir, 'char_to_int.pkl')
    with open(char_to_int_path, 'rb') as f:
        char_to_int = pickle.load(f)
        f.close()
    return char_to_int


def get_one_hot(sentence, char_to_int):
    one_hot = []
    for j in range(config.text_seq_length):
        if j >= len(sentence):
            one_hot.append(char_to_int['pad'])
        elif sentence[j] not in char_to_int:
            one_hot.append(char_to_int['unk'])
        else:
            one_hot.append(char_to_int[sentence[j]])
    return np.array(one_hot)


def extract_noisy_correct_label(question_label_data):
    noisy_sents, correct_sents, labels = [], [], []
    char_to_int = load_vocab()
    for idx, question_label in enumerate(question_label_data):
        if len(question_label) > 0:
            noisy_sent, correct_sent, label = question_label.split("\t\t")
            noisy_sents.append(get_one_hot(noisy_sent, char_to_int))
            correct_sents.append(get_one_hot(correct_sent, char_to_int))
            labels.append(label)
    noisy_sents = np.array(noisy_sents)
    correct_sents = np.array(correct_sents)
    labels = np.array(labels)
    return noisy_sents, correct_sents, labels


def extract_label(label_strings):
    processed_labels = np.ones((len(label_strings), config.label_seq_length, 2)) * -1
    for label_seq_length, labels in enumerate(label_strings):
        labels = labels.strip().split("\t")
        for seq_idx, label in enumerate(labels):
            if seq_idx < config.label_seq_length:
                label_split = label.split(" ")
                entity = int(label_split[0])
                entity_instance = int(label_split[1])
                processed_labels[label_seq_length, seq_idx] = np.array([entity, entity_instance])
    return processed_labels


def get_incorrect_indices(indices, low, high):
    incorrect_indices = []
    for index in indices:
        r = np.random.randint(low, high)
        while r == index:
            r = np.random.randint(low, high)
        incorrect_indices.append(r)
    return np.array(incorrect_indices)


def get_batch_data(noisy_sents, correct_sents, labels, batch_size, random=True, start=0):
    if random:
        indices = np.random.randint(0, len(noisy_sents), batch_size)
        incorrect_indices = get_incorrect_indices(indices, 0, len(noisy_sents))
    else:
        indices = range(start, min(start + batch_size, len(noisy_sents)))
        incorrect_indices = get_incorrect_indices(indices, 0, len(noisy_sents))
    noisy_sents = np.array(noisy_sents)
    correct_sents = np.array(correct_sents)
    labels = np.array(labels)
    return noisy_sents[indices], correct_sents[indices], correct_sents[incorrect_indices],\
           labels[indices], labels[incorrect_indices]


def split_train_valid(noisy_sents, correct_sents, labels, split=0.8):
    indices = np.random.choice(len(noisy_sents), size=len(noisy_sents), replace=False)
    boundary = int(split * len(noisy_sents))
    train_indices = indices[:boundary]
    valid_indices = indices[boundary:]
    train_set = noisy_sents[train_indices], correct_sents[train_indices], labels[train_indices]
    valid_set = noisy_sents[valid_indices], correct_sents[valid_indices], labels[valid_indices]
    return train_set, valid_set


def get_batch_data_iterator(n_epoch, question_label_data, batch_size, split=0.8):
    noisy_sents, correct_sents, label_strings = extract_noisy_correct_label(question_label_data)
    num_batches_per_epoch = int(len(noisy_sents) / batch_size)
    del question_label_data
    labels = extract_label(label_strings)
    del label_strings
    train_set, valid_set = split_train_valid(noisy_sents, correct_sents, labels, split)
    del noisy_sents, correct_sents, labels
    noisy_sents_train = train_set[0]
    correct_sents_train = train_set[1]
    labels_train = train_set[2]
    noisy_sents_valid = valid_set[0]
    correct_sents_valid = valid_set[1]
    labels_valid = valid_set[2]
    for i in range(n_epoch):
        for j in range(num_batches_per_epoch):
            train_batch = get_batch_data(noisy_sents_train, correct_sents_train, labels_train, batch_size=batch_size)
            valid_batch = get_batch_data(noisy_sents_valid, correct_sents_valid, labels_valid, batch_size=batch_size)
            yield train_batch, valid_batch


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), os.path.pardir, 'data')
    incorrect_correct_label_sent_path = os.path.join(data_dir, 'processed_data',
                                                     'incorrect_correct_label_sent_path.txt')
    with open(incorrect_correct_label_sent_path, 'rb') as f:
        incorrect_correct_label_sent_data = f.readlines()
        f.close()
    batch_iterator = get_batch_data_iterator(1, incorrect_correct_label_sent_data, 64)
    for i in range(10):
        train_batch, valid_batch = batch_iterator.next()
