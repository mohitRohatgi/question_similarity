import os
import numpy as np
import pickle

from config.config import Config


def construct_embedding_and_vocab():
    resources_dir = os.path.join(os.getcwd(), os.pardir, 'resources')
    embedding_path = os.path.join(resources_dir, 'glove.840B.300d-char.txt')
    embed_matrix_path = os.path.join(resources_dir, 'embedding_matrix.npy')
    char_to_int_path = os.path.join(resources_dir, 'char_to_int.pkl')
    int_to_char_path = os.path.join(resources_dir, 'int_to_char.pkl')

    if not os.path.exists(resources_dir):
        os.mkdir(resources_dir)

    if not os.path.exists(embed_matrix_path):
        config = Config()
        embed_matrix = [np.zeros(config.embeddings_dim), np.ones(config.embeddings_dim) * -1]
        char_to_int = dict()
        int_to_char = dict()
        char_to_int['pad'] = 0
        char_to_int['unk'] = 1
        with open(embedding_path, 'rb') as f:
            lines = f.readlines()
            f.close()
        for index, line in enumerate(lines):
            line = line.split(" ")
            char_to_int[line[0]] = index + 2
            int_to_char[index+1] = line[0]
            embed = np.array(line[1:]).astype(np.float)
            embed_matrix.append(embed)
        embed_matrix = np.array(embed_matrix)
        np.save(embed_matrix_path, embed_matrix)
        with open(int_to_char_path, 'wb') as f:
            pickle.dump(int_to_char, f)
            f.close()
        with open(char_to_int_path, 'wb') as f:
            pickle.dump(char_to_int, f)
            f.close()


if __name__ == '__main__':
    construct_embedding_and_vocab()