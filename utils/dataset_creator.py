import os
import json
import numpy as np
import pickle
import re
from utils.preprocessor import get_one_hot, load_vocab

# format of the dataset assumed is noisy sentence \t\t correct sentence \t\t label \n
# label -> label instance \t label instance \t ....
# label instance -> entity space entity-instance

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ]


def noise_maker(sentence, threshold, num_dirt_sentences):
    noisy_sentences = [sentence]
    for i in range(num_dirt_sentences):
        noisy_sentence = []
        i = 0
        while i < len(sentence):
            random = np.random.uniform()
            if random < threshold:
                noisy_sentence.append(sentence[i])
            else:
                new_random = np.random.uniform()
                if new_random > 0.67:
                    if i == len(sentence) - 1:
                        continue
                    else:
                        noisy_sentence.append(sentence[i + 1])
                        noisy_sentence.append(sentence[i])
                        i += 1
                elif new_random < 0.33:
                    random_letter = np.random.choice(letters, 1)[0]
                    noisy_sentence.append(random_letter)
                    noisy_sentence.append(sentence[i])
                else:
                    pass
            i += 1
        noisy_sentence = ''.join(noisy_sentence)
        noisy_sentences.append(noisy_sentence)
    return noisy_sentences


def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]', '', text)
    text = re.sub('a0', '', text)
    text = re.sub('\'92t', '\'t', text)
    text = re.sub('\'92s', '\'s', text)
    text = re.sub('\'92m', '\'m', text)
    text = re.sub('\'92ll', '\'ll', text)
    text = re.sub('\'91', '', text)
    text = re.sub('\'92', '', text)
    text = re.sub('\'93', '', text)
    text = re.sub('\'94', '', text)
    text = re.sub('\.', '. ', text)
    text = re.sub('\!', '! ', text)
    text = re.sub('\?', '? ', text)
    text = re.sub(' +', ' ', text)
    return text


# format of the final dataset to produced is 'spell-mistakes-questions \t correct-questions \t labels'
# entity is the filename and entity instances are instances contained in it.
def sort_text(corpus_lines):
    sorted_corpus_lines = []
    min_length = min([len(line) for line in corpus_lines])
    max_length = max([len(line) for line in corpus_lines])
    for i in range(min_length, max_length + 1):
        for sentence in corpus_lines:
            if len(sentence) == i:
                sorted_corpus_lines.append(re.sub(' +', ' ', sentence))
    return sorted_corpus_lines


def create_question_label_data(question_label_path, question_corpus_path, entity_instance2label):
    question_label_data = ""
    with open(question_corpus_path, 'rb') as corpus:
        corpus_lines = corpus.readlines()
        corpus.close()
    corpus_lines = sort_text(corpus_lines)
    for line in corpus_lines:
        line = line.strip().lower()
        if len(line) > 10 and not ('cycle' in line or 'lex' in line):
            labels = ""
            for i in range(len(line)):
                for j in range(i + 1, len(line)):
                    entity_instance = line[i:j]
                    if entity_instance in entity_instance2label:
                        label = entity_instance2label[entity_instance]
                        labels += str(label[0]) + " " + str(label[1]) + "\t"
            if len(labels.strip()) > 0:
                question_label_data += line.strip() + "\t\t" + labels.strip() + "\n"
    question_label_data = question_label_data.strip()
    with open(question_label_path, 'wb') as f:
        f.write(question_label_data)
        f.close()
    return question_label_data.split('\n')


def create_incorrect_correct_label_data(question_label_data, incorrect_correct_sent_path):
    incorrect_correct_sent_data = ""
    for line in question_label_data:
        question, label_string = line.split("\t\t")
        noisy_questions = noise_maker(question, 0.8, 3)
        for noisy_question in noisy_questions:
            incorrect_correct_sent_data += noisy_question.strip() + " \t\t" + question.strip() + "\t\t" + label_string.strip() + "\n"
    with open(incorrect_correct_sent_path, 'wb') as incorrect_correct_sent_file:
        incorrect_correct_sent_file.write(incorrect_correct_sent_data)
        incorrect_correct_sent_file.close()
    return incorrect_correct_sent_data


# entities_instance2label contains given a entity instance find the label
# each entity label contains two indices -> one for entity index and
# the other for entity instance in the entity file
def create_entities_dict(entities_dir, entities_instance2label_path, entity_idx2entity_path):
    entity_idx2entity = dict()
    entities_instance2label = dict()
    for entity_idx, file_path in enumerate(os.listdir(entities_dir)):
        entity = file_path.split(".")[0]
        entity_idx2entity[entity_idx] = entity
        with open(os.path.join(entities_dir, file_path), 'r') as entity_file:
            entity_json_list = json.load(entity_file)
            for entity_instance_idx, entity_json in enumerate(entity_json_list):
                label = np.array([entity_idx, entity_instance_idx])
                value = entity_json['value'].strip().lower()
                synonyms = entity_json['synonyms']
                entities_instance2label[value] = label
                for synonym in synonyms:
                    synonym = synonym.strip().lower()
                    entities_instance2label[synonym] = label
    with open(entities_instance2label_path, 'wb') as entity_file:
        pickle.dump(entities_instance2label, entity_file)
        entity_file.close()
    with open(entity_idx2entity_path, 'wb') as entity_idx_file:
        pickle.dump(entity_idx2entity, entity_idx_file)
        entity_idx_file.close()
    return entities_instance2label, entity_idx2entity


def get_question_vocabs(question_label_path):
    with open(question_label_path, 'r') as f:
        question_labels = f.readlines()
        f.close()
    question_idx2label = dict()
    label2_question_indices = dict()
    question2question_idx = dict()
    question_idx2question = dict()
    sents_train = []
    char_to_int = load_vocab()
    for idx, question_label in enumerate(question_labels):
        question, label = question_label.split("\t\t")
        sents_train.append(get_one_hot(question, char_to_int))
        question_idx2label[idx] = label
        question2question_idx[question] = idx
        question_idx2question[idx] = question
        if label in label2_question_indices:
            label2_question_indices[label].append(idx)
        else:
            label2_question_indices[label] = [idx]
    sents_train = np.array(sents_train)
    return question_idx2label, question2question_idx, question_idx2question, label2_question_indices, sents_train


def main():
    data_dir = os.path.join(os.getcwd(), os.path.pardir, 'data')
    entities_dir = os.path.join(data_dir, 'entities')
    question_corpus = os.path.join(data_dir, 'Question_Corpus.txt')
    processed_data = os.path.join(data_dir, 'processed_data')
    question_label_path = os.path.join(processed_data, 'question_label_data.txt')
    correct_incorrect_label_path = os.path.join(processed_data, 'correct_incorrect_label_data.txt')
    # entities dict maps entities to label.
    entities_instance2label_path = os.path.join(processed_data, 'entities_instance2label.pkl')
    incorrect_correct_label_sent_path = os.path.join(processed_data, 'incorrect_correct_label_sent_path.txt')
    entities_instance2label = None
    question_label_data = None
    entity_idx2entity_path = os.path.join(processed_data, 'entity_idx2entity.pkl')

    if not os.path.exists(processed_data):
        os.mkdir(processed_data)

    # entities dict contains given a string find the label
    if not os.path.isfile(entities_instance2label_path):
        entities_instance2label, _ = create_entities_dict(entities_dir, entities_instance2label_path,
                                                          entity_idx2entity_path)

    if entities_instance2label is None:
        with open(entities_instance2label_path, 'r') as entities_instance2label_file:
            entities_instance2label = pickle.load(entities_instance2label_file)

    if not os.path.exists(processed_data):
        os.makedirs(processed_data)

    if not os.path.isfile(question_label_path):
        question_label_data = create_question_label_data(question_label_path, question_corpus, entities_instance2label)

    if question_label_data is None:
        with open(question_label_path, 'rb') as f:
            question_label_data = f.readlines()
            f.close()

    if not os.path.isfile(correct_incorrect_label_path):
        create_incorrect_correct_label_data(question_label_data, incorrect_correct_label_sent_path)


if __name__ == '__main__':
    main()
