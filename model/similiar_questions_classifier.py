import tensorflow as tf
import numpy as np

from config.config import Config


class SimilarQuestionClassifier:
    def __init__(self):
        self.config = Config()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.add_model()

    def initialise(self, sess, embed_matrix):
        sess.run(tf.global_variables_initializer())
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embed_matrix})

    def add_model(self):
        self.add_placeholders()
        self.add_embeddings()
        self.assemble_model()
        self.add_loss()
        self.add_train_op()

    def add_placeholders(self):
        self.noisy_sent = tf.placeholder(dtype=tf.int32, shape=(None, self.config.text_seq_length),
                                         name='noisy_sent')
        self.correct_sent = tf.placeholder(dtype=tf.int32, shape=(None, self.config.text_seq_length),
                                           name='correct_sent')
        self.label = tf.placeholder(dtype=tf.int32, shape=(None, self.config.label_seq_length, 2))
        # for negative sampling
        self.incorrect_sent = tf.placeholder(dtype=tf.int32, shape=(None, self.config.text_seq_length),
                                             name='correct_sent')
        self.incorrect_label = tf.placeholder(dtype=tf.int32, shape=(None, self.config.label_seq_length, 2))
        self.embedding_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.config.vocab_size,
                                                                           self.config.embeddings_dim))
        self.dropout_keep = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_keep')

    def assemble_model(self):
        self.add_ner_model()
        self.add_question_similarity_model()

    def add_embeddings(self):
        with tf.variable_scope("embeddings"):
            self.char_embedding = tf.get_variable(shape=[self.config.vocab_size, self.config.embeddings_dim],
                                                  trainable=False, name='embeddings', dtype=tf.float32)
            self.embedding_init = self.char_embedding.assign(self.embedding_placeholder)
            noisy_sent = tf.nn.embedding_lookup(self.char_embedding, self.noisy_sent)
            self.noisy_sent_vector = tf.cast(noisy_sent, tf.float32)
            correct_sent = tf.nn.embedding_lookup(self.char_embedding, self.correct_sent)
            self.correct_sent_vector = tf.cast(correct_sent, tf.float32)
            incorrect_sent = tf.nn.embedding_lookup(self.char_embedding, self.incorrect_sent)
            self.incorrect_sent_vector = tf.cast(incorrect_sent, tf.float32)

    def create_lstm_multicell(self, hidden_dim, n_layer):
        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(hidden_dim, reuse=tf.get_variable_scope().reuse)

        lstm_multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layer)])
        return lstm_multi_cell

    # ner_output = (batch_size, label_seq_length, 2)
    def add_ner_model(self):
        with tf.variable_scope("ner"):
            noisy_sent_lstm = self.create_lstm_multicell(self.config.num_hidden, self.config.lstm_layers)
            noisy_sent_lstm_initial_state = noisy_sent_lstm.zero_state(tf.shape(self.noisy_sent)[0], tf.float32)
            _, ner_final_states = \
                tf.nn.dynamic_rnn(noisy_sent_lstm, self.noisy_sent_vector, initial_state=noisy_sent_lstm_initial_state)
            entity_U = tf.get_variable('entity_U', shape=(self.config.num_hidden, self.config.label_seq_length),
                                       dtype=tf.float32)
            entity_B = tf.get_variable('entity_B', shape=self.config.label_seq_length, dtype=tf.float32)
            ner_entity_output = tf.add(tf.matmul(ner_final_states[-1].h, entity_U), entity_B)
            ner_entity_instance_input = tf.concat([ner_final_states[-1].h, ner_entity_output], axis=-1)
            enitity_instance_U = tf.get_variable(name='enitity_instance_U', dtype=tf.float32,
                                                 shape=(self.config.num_hidden + self.config.label_seq_length,
                                                        self.config.label_seq_length))
            enitity_instance_B = tf.get_variable(name='enitity_instance_B', dtype=tf.float32,
                                                 shape=(self.config.label_seq_length, ))
            ner_entity_instance_output = tf.add(tf.matmul(ner_entity_instance_input, enitity_instance_U),
                                                enitity_instance_B)
            self.ner_output = tf.concat([tf.expand_dims(ner_entity_output, axis=-1),
                                         tf.expand_dims(ner_entity_instance_output, axis=-1)],
                                        axis=-1)

    def add_question_similarity_model(self):
        with tf.variable_scope("question_similarity", reuse=tf.AUTO_REUSE):
            self.noisy_question_final_state = self.get_score_from_question_encoder('noisy_question',
                                                                                   self.noisy_sent_vector,
                                                                                   self.ner_output)
            correct_label = tf.cast(self.label, dtype=tf.float32)
            self.correct_question_final_state = self.get_score_from_question_encoder('correct_question',
                                                                                     self.correct_sent_vector,
                                                                                     correct_label)
            incorrect_label = tf.cast(self.incorrect_label, dtype=tf.float32)
            self.incorrect_question_final_state = self.get_score_from_question_encoder('correct_question',
                                                                                       self.incorrect_sent_vector,
                                                                                       incorrect_label)
            # Taking dot product of two vector -> range=(-1, 1)
            self.correct_question_score = tf.reduce_sum(self.noisy_question_final_state *
                                                        self.correct_question_final_state,
                                                        axis=-1)
            self.correct_question_score /= tf.sqrt(tf.reduce_sum(tf.square(self.noisy_question_final_state), axis=-1))
            self.correct_question_score /= tf.sqrt(tf.reduce_sum(tf.square(self.correct_question_final_state), axis=-1))
            self.incorrect_question_score = tf.reduce_sum(self.noisy_question_final_state *
                                                          self.incorrect_question_final_state,
                                                          axis=-1)
            self.incorrect_question_score /= tf.sqrt(tf.reduce_sum(tf.square(self.noisy_question_final_state), axis=-1))
            self.incorrect_question_score /= tf.sqrt(tf.reduce_sum(tf.square(self.incorrect_question_final_state),
                                                                   axis=-1))
            # transforming score to have range = (0, 1)
            self.correct_question_score = (self.correct_question_score + 1.0) / 2.0
            self.incorrect_question_score = (self.incorrect_question_score + 1.0) / 2.0

    def get_score_from_question_encoder(self, scope, input_vec, ner_labels):
        with tf.variable_scope(scope):
            ner_output = tf.reshape(ner_labels, shape=(-1, 1, self.config.label_seq_length * 2))
            ner_output_repeated = tf.concat([ner_output for _ in range(self.config.text_seq_length)], axis=1)
            question_input = tf.concat([input_vec, ner_output_repeated], axis=-1)
            question_lstm = self.create_lstm_multicell(self.config.num_hidden, self.config.lstm_layers)
            question_lstm_initial_state = question_lstm.zero_state(tf.shape(question_input)[0], tf.float32)
            _, question_encoder_final_states = tf.nn.dynamic_rnn(question_lstm, question_input,
                                                                 initial_state=question_lstm_initial_state)
            return question_encoder_final_states[-1].h

    def add_loss(self):
        # negative sampling loss
        delta = 1e-12
        self.loss = tf.reduce_sum(tf.maximum(-tf.log(self.correct_question_score + delta) +
                                  tf.log(self.incorrect_question_score + delta), 0))

    # noisy_final_states = (batch_size, num_hidden)
    # correct_final_state = (batch_size, num_hidden)
    def get_accuracy(self, noisy_final_states, correct_final_states):
        correct_predictions = np.zeros(len(noisy_final_states))
        for index, noisy_final_state in enumerate(noisy_final_states):
            dot_prod = np.sum(noisy_final_state * correct_final_states, axis=-1)
            dot_prod /= np.sqrt(np.sum(np.square(correct_final_states), axis=-1))
            dot_prod /= np.sqrt(np.sum(np.square(noisy_final_state)))
            if index == np.argmax(dot_prod):
                correct_predictions[index] = 1
        return np.mean(correct_predictions)

    def add_train_op(self):
        with tf.variable_scope("training"):
            grads, variables = zip(*self.optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(zip(grads, variables))

    def run_batch(self, sess, noisy_sents_train, correct_sents_train,
                  incorrect_sents_train, labels_train, incorrect_labels_train, is_train=True):
        feed_dict = {
            self.noisy_sent: noisy_sents_train,
            self.correct_sent: correct_sents_train,
            self.label: labels_train,
            self.incorrect_sent: incorrect_sents_train,
            self.incorrect_label: incorrect_labels_train,
            self.dropout_keep: self.config.dropout_keep
        }
        fetch = [self.loss, self.noisy_question_final_state, self.correct_question_final_state,
                 self.correct_question_score, self.incorrect_question_score]
        if is_train:
            fetch.append(self.train_op)
            loss, noisy_question_final_state, correct_question_final_state, cor, incor, _ = sess.run(fetch, feed_dict)
        else:
            loss, noisy_question_final_state, correct_question_final_state, cor, incor = sess.run(fetch, feed_dict)
        accuracy = self.get_accuracy(noisy_question_final_state, correct_question_final_state)
        return loss, accuracy
