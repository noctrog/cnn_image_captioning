import argparse
import googletrans
import json

def main(args):

    vocab = set()
    with open(args.file, "r") as captions_file:
        data = json.load(captions_file)
        for i in range(len(data['annotations'])):
            words = data['annotations'][i]['caption'].split(' ')
            for word in words:
                vocab.add(word)

    print(len(vocab))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='', help='JSON que contiene los captions')
    args = parser.parse_args()

    assert args.file != ''

    main(args)
