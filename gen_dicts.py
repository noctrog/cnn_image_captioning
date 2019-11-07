import argparse
import json
import re

import torch
from spellchecker import SpellChecker

def main(args):

    spell = SpellChecker(distance=1)

    print('Leyendo palabras...')
    word_set = set()
    with open(args.file, "r") as captions_file:
        data = json.load(captions_file)
        for i in range(len(data['annotations'])):
            words = re.findall(r"[\w']+|[.,!?;]", data['annotations'][i]['caption'])
            for word in words:
                word_set.add(word)

    print('Corrigiendo faltas...')
    correct_word_set = set([spell.correction(word).lower() for word in word_set])
    print(len(word_set))
    print(len(correct_word_set))
    word_list = list(correct_word_set)

    print('Guardando diccionarios...')
    ids = [i for i in range(len(word_list))]
    # Generar stoi
    stoi = dict(zip(word_list, ids))
    # Generar itos
    itos = dict(zip(ids, word_list))

    # Generar principio y fin de frases
    stoi['<s>'] = len(stoi)
    stoi['</s>'] = len(stoi)
    itos[len(itos)] = '<s>'
    itos[len(itos)] = '</s>'

    # Guardar diccionarios
    dicts = {'stoi': stoi,
             'itos': itos}
    torch.save(dicts, './weights/dicts')
    print('Hecho.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='', help='JSON que contiene los captions')
    args = parser.parse_args()

    assert args.file != ''

    main(args)
