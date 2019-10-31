import argparse
import googletrans
import json
import torch

from torchtext import vocab

import mini_glove

def main(args):

    # Vocabulario original de GloVe, 400000 palabras
    glove = vocab.GloVe(name='6B', dim=300)

    word_set = set()
    with open(args.file, "r") as captions_file:
        data = json.load(captions_file)
        for i in range(len(data['annotations'])):
            words = data['annotations'][i]['caption'].lower() \
                .replace('.', ' . ').replace(',', ' , ').replace('\'', ' \' ') \
                .replace('(', ' ( ').replace(')', ') ').replace('-', ' - ') \
                .replace('"', ' " ').replace(':', ' : ').replace(';', ' ; ') \
                .replace('!', ' ! ').replace('?', ' ? ').split(' ')
            for word in words:
                word_set.add(word)

    word_list = list(word_set)

    glove_words = [word for word in word_list if word in glove.stoi]
    glove_ids = [glove.stoi[word] for word in glove_words]
    vectors = glove.vectors[glove_ids]
    count = 0
    for word in word_list:
        if word not in glove.stoi:
            count += 1

    # Generar stoi
    new_ids = [i for i in range(len(glove_words))]
    stoi = dict(zip(glove_words, new_ids))

    # Generar itos
    itos = dict(zip(new_ids, glove_words))

    # Generar miniglove y guardarlo
    miniglove = mini_glove.MiniGlove(vectors, stoi, itos)

    # Aniadir palabra equivalente a <unknown> y <start>
    miniglove.stoi['<unknown>'] = len(miniglove.stoi)
    miniglove.stoi['<s>'] = len(miniglove.stoi)
    miniglove.itos[len(miniglove.itos)] = '<unknown>'
    miniglove.itos[len(miniglove.itos)] = '<s>'
    # Inserta <unknown>
    miniglove.vectors = torch.cat([miniglove.vectors, glove.vectors[glove.stoi['sandberger']].view(1, 300)], dim=0)
    # Inserta <s>
    miniglove.vectors = torch.cat([miniglove.vectors, torch.ones([1, 300])], dim=0)

    miniglove.save()

    print('Palabras leidas: ' + str(len(word_set)))
    print('Palabras guardadas: ' + str(vectors.shape))
    print('Palabras no guardadas: ' + str(count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='', help='JSON que contiene los captions')
    parser.add_argument("-o", type=str, default='./weights/glove_striped.dat')
    args = parser.parse_args()

    assert args.file != ''

    main(args)
