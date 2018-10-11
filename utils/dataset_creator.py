import os
import json
import numpy as np
import pickle


# format of the final dataset to produced is 'spell-mistakes-questions \t correct-questions \t labels'
# entity is the filename and entity instances are instances contained in it.
def create_question_label_map(question_label_path, question_corpus_path, entity_instance2label):
    question_label_data = ""
    with open(question_corpus_path, 'rb') as corpus:
        for line in corpus.readlines():
            labels = ""
            for i in range(len(line)):
                for j in range(i + 1, len(line)):
                    entity_instance = line[i:j]
                    if entity_instance in entity_instance2label:
                        label = entity_instance2label[entity_instance]
                        labels += str(label[0]) + " " + str(label[1]) + "\t"
            question_label_data += line.strip() + "\t\t" + labels.strip() + "\n"
        corpus.close()
    with open(question_label_path, 'wb') as f:
        f.write(question_label_data)
        f.close()
    return question_label_data


def create_incorrect_correct_label_data(question_label_data, incorrect_correct_sent_path):
    pass


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


def create_incorrect_correct_sent_data(question_label_path):
    pass


def main():
    data_dir = os.path.join(os.getcwd(), os.path.pardir, 'data')
    entities_dir = os.path.join(data_dir, 'entities')
    question_corpus = os.path.join(data_dir, 'Question_Corpus.txt')
    processed_data = os.path.join(data_dir, 'processed_data')
    question_label_path = os.path.join(processed_data, 'question_label_data.txt')
    correct_incorrect_label_path = os.path.join(processed_data, 'correct_incorrect_label_data.txt')
    # entities dict maps entities to label.
    entities_instance2label_path = os.path.join(processed_data, 'entities_instance2label.pkl')
    incorrect_correct_sent_path = os.path.join(processed_data, 'incorrect_correct_sent.txt')
    entities_instance2label = None
    question_label_data = None
    entity_idx2entity_path = os.path.join(processed_data, 'entity_idx2entity.pkl')

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
        create_question_label_map(question_label_path, question_corpus, entities_instance2label)

    if not os.path.isfile(correct_incorrect_label_path) is None:
        create_incorrect_correct_sent_data(question_label_path)

    if not os.path.isfile(correct_incorrect_label_path):
        create_incorrect_correct_label_data(question_label_data, incorrect_correct_sent_path)


if __name__ == '__main__':
    main()
