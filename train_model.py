import os
import tensorflow as tf
import numpy as np

from config.config import Config
from model.similiar_questions_classifier import SimilarQuestionClassifier
from utils.preprocessor import get_batch_data_iterator


def load_embed_matrix():
    embedding_matrix_path = os.path.join(os.getcwd(), 'resources', 'embedding_matrix.npy')
    return np.squeeze(np.load(embedding_matrix_path))


def train():
    config = Config()
    data_dir = os.path.join(os.getcwd(), 'data')
    incorrect_correct_label_sent_path = os.path.join(data_dir, 'processed_data',
                                                     'incorrect_correct_label_sent_path.txt')
    with open(incorrect_correct_label_sent_path, 'rb') as f:
        incorrect_correct_label_sent_data = f.readlines()
        f.close()
    num_batches_per_epoch = int(len(incorrect_correct_label_sent_data) * config.train_valid_split / config.batch_size)
    batch_iterator = get_batch_data_iterator(config.n_epoch, incorrect_correct_label_sent_data, config.batch_size,
                                             config.train_valid_split)

    embedding_matrix = load_embed_matrix()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = SimilarQuestionClassifier()
            model.initialise(sess, embedding_matrix)
            for i in range(config.n_epoch):
                avg_loss = 0.0
                avg_accuracy = 0.0
                for j in range(num_batches_per_epoch):
                    train_batch, valid_batch = batch_iterator.next()
                    noisy_sents_train, correct_sents_train, incorrect_sents_train, labels_train, incorrect_labels_train\
                        = train_batch
                    noisy_sents_valid, correct_sents_valid, labels_valid, incorrect_sents_valid, incorrect_labels_valid\
                        = valid_batch
                    loss, accuracy = model.run_batch(sess, noisy_sents_train, correct_sents_train,
                                                     incorrect_sents_train, labels_train, incorrect_labels_train)
                    avg_loss += loss
                    avg_accuracy += accuracy

                    print("epoch = ", i, " batch_num = ", j, " loss = ", loss, " accuracy = ", accuracy)

                    if j % config.evaluate_every == 0:
                        loss, accuracy = model.run_batch(sess, noisy_sents_valid, correct_sents_valid,
                                                         labels_valid, incorrect_sents_valid,
                                                         incorrect_labels_valid)
                        print("epoch = ", i, " batch_num = ", j, " valid_loss = ", loss, " valid_accuracy = ", accuracy)
                print("epoch = ", i, " avg_loss = ", avg_loss / num_batches_per_epoch,
                      " avg_accuracy = ", avg_accuracy / num_batches_per_epoch)


if __name__ == '__main__':
    train()
