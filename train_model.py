import os
import tensorflow as tf

from config.config import Config
from model.similiar_questions_classifier import SimilarQuestionClassifier
from utils.preprocessor import get_batch_data_iterator


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
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = SimilarQuestionClassifier()
            for i in range(config.n_epoch):
                for j in range(num_batches_per_epoch):
                    train_batch, valid_batch = batch_iterator.next()
                    noisy_sents_train, correct_sents_train, labels_train = train_batch
                    noisy_sents_valid, correct_sents_valid, labels_valid = valid_batch
                    # loss, accuracy = model.run_batch(sess, noisy_sents_train, correct_sents_train, labels_train)
                    #
                    # print("epoch = ", i, " batch_num = ", j, " loss = ", loss, " accuracy = ", accuracy)
                    #
                    # if j % config.evaluate_every == 0:
                    #     loss, accuracy = model.run_batch(sess, noisy_sents_train, correct_sents_train,
                    #                                      labels_train, is_valid=True)
                    #     print("epoch = ", i, " batch_num = ", j, " loss = ", loss, " accuracy = ", accuracy)


if __name__ == '__main__':
    train()