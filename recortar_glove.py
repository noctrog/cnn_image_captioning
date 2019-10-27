import argparse
import googletrans
import json

from torchtext import vocab

import mini_glove

def main(args):

    # Vocabulario original de GloVe, 400000 palabras
    glove = vocab.GloVe(name='6B', dim=300)

    word_set = set()
    with open(args.file, "r") as captions_file:
        data = json.load(captions_file)
        for i in range(len(data['annotations'])):
            words = data['annotations'][i]['caption'].lower().replace('.', ' .') .replace(',', ' ,') .replace('\'', ' \'').split(' ')
            for word in words:
                word_set.add(word)

    word_list = list(word_set)
    vectors = glove.vectors[[glove.stoi[word] for word in word_list if word in glove.stoi]]

    # TODO: Generar stoi

    # TODO: Generar itos

    # Generar miniglove y guardarlo
    miniglove = mini_glove.MiniGlove()

    print(len(word_set))
    print(vectors.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='', help='JSON que contiene los captions')
    parser.add_argument("-o", type=str, default='./weights/glove_striped.dat')
    args = parser.parse_args()

    assert args.file != ''

    main(args)
