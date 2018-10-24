import os
import numpy as np
import tensorflow as tf
from config.config import Config
from utils.dataset_creator import get_question_vocabs
from utils.preprocessor import load_vocab, get_one_hot
from model.similiar_questions_classifier import SimilarQuestionClassifier


def get_similiar_question_indices(test_sent_rep, sents_reps, question_idx2label, label2_question_indices):
    best_match = -1
    best_match_dot_prod = -2
    for idx, sents_rep in enumerate(sents_reps):
        dot_prod = np.sum(sents_rep * test_sent_rep) / np.sqrt(np.sum(np.square(sents_reps)))
        dot_prod /= np.sqrt(np.sum(np.square(test_sent_rep)))
        if best_match_dot_prod < dot_prod:
            best_match_dot_prod = dot_prod
            best_match = idx
    return label2_question_indices[question_idx2label[best_match]]


def test():
    config = Config()
    char_to_int = load_vocab()
    saved_question_representation_path = os.path.join(os.getcwd(), 'saved_models', 'tf_model_graphs')
    data_dir = os.path.join(os.getcwd(), 'data')
    processed_data = os.path.join(data_dir, 'processed_data')
    question_idx2label_path = os.path.join(processed_data, 'question_idx2label.pkl')
    label2question_indices_path = os.path.join(processed_data, 'label2question_indices.pkl')
    question_label_path = os.path.join(processed_data, 'question_label_data.txt')
    question_idx2label, question2question_idx, question_idx2question, label2_question_indices, sents_train = \
        get_question_vocabs(question_label_path)

    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            model = SimilarQuestionClassifier()
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(saved_question_representation_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            sents_reps = model.get_rep(sess, sents_train)
            while True:
                test_question = raw_input("Please enter the test question here:  ")
                test_sent_rep = model.get_rep(sess, np.expand_dims(get_one_hot(test_question, char_to_int), axis=0))
                print('Similiar Questions found by model are below:')
                indices = get_similiar_question_indices(test_sent_rep[0], sents_reps, question_idx2label,
                                                        label2_question_indices)
                responses = []
                for idx in indices:
                    responses.append(question_idx2question[idx])
                print(responses)


    # if not os.path.exists(saved_question_representation_path):
    #     find_question_rep(saved_question_representation_path, model)
    # questions_rep = load_rep(saved_question_representation_path)


if __name__ == '__main__':
    test()
