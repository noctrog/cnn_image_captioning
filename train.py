import argparse
import os.path as path

import numpy as np

import torch
from torch import optim
from torchvision import datasets, transforms

import model

from tensorboardX import SummaryWriter

## ------------------------------------------------------------
## ------------------------------------------------------------
## -------------- Leer los datos y procesarlos ----------------

# Las imagenes tienen que ir de 0 a 1, con un tamanio de 224x224
# despues normalizadas con normalize = 
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 
# 0.224, 0.225]) porque la red convolucional se entreno con 
# imagenes con este formato

#TODO
def dataloader(image_folder, captions_file, batch_size):

    yield False


## ------------------------------------------------------------
## ------------------------------------------------------------
## -------------- Bucle de entrenamiento ----------------------

def main(args):

    # Usar CUDA si esta disponible
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # Mantiene un tensorboard con datos actualizados del entrenamiento
    writer = SummaryWriter(comment='CNN_RNN')

    # Crea los modelos a entrenar
    if path.exists('weights/encoder.dat'):
        dump = torch.load('weights/encoder.dat')
        encoder = model.Encoder(dump['latent_size']).to(device)
        encoder.load_state_dict(dump['state_dict'])
    else:
        encoder = model.Encoder(latent_size=1024).to(device)
    if path.exists('weights/decoder.dat'):
        dump = torch.load('weights/decoder.dat')
        decoder = model.Decoder(dump['latent_size'], dump['hidden_size'], dump['vocab_size'],
                                dump['max_seq_length'], dump['dropout'], dump['n_layers']) .to(device)
        decoder.load_state_dict(dump['state_dict'])
    else:
        decoder = model.Decoder(latent_size=1024, hidden_size=1024, vocab_size=10, # TODO vocab_size!!
                                max_seq_length=20)

    # algoritmo que actualizara los pesos de las redes
    parameters = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = optim.SGD(parameters, lr=args.lr)

    # Funcion de perdida a usar: Entropia cruzada ya que los elementos a predecir (palabras)
    # son mutuamente exlusivos (solo se puede elegir una palabra)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(args.epochs):
        # Lee las imagenes con una frase distinta cada vez
        trainloader = dataloader(args.image_folder, args.captions_file, args.batch_size)

        losses = []
        for i, images, captions, lengths in enumerate(trainloader):
            # Lee el siguiente batch
            images_v = images.to(device)
            captions_v = captions.to(device)
            lengths_v = lengths.to(device)

            # Limpia los optimizadores
            optimizer.zero_grad()

            # Convierte las imagenes a su representacion en el espacio latente
            latent_v = encoder(images_v)

            # Pasa los puntos del espacio latente por la red recurrente
            outputs = decoder(latent_v, captions_v, lengths_v)

            # Pasar salida esperada a one-hot
            expected_v = torch.nn.functional.one_hot(captions_v, decoder.vocab_size)

            # Calcula la pérdida para actualizar las redes
            loss = criterion(outputs, expected_v)

            # Actuaslizar los pesos 
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Informar sobre el estado del entrenamiento
            if i % 100 and i != 0:
                # Actualizar tensorboard
                writer.add_scalar('Training loss', np.mean(losses[-100:]))

                # Imprimir status
                print('Epoch: {}\tBatch: {}\t\tTraining loss:', e, i, np.mean(losses[-100:]))
    else:
        print('-------- Epoch: {}\t\tLoss: {} -----------', e, np.mean(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.002, help='Learning rate')
    parser.add_argument("--epochs", type=int, default=1, help="Numero de generaciones a entrenar")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch")
    parser.add_argument("--image_folder", type=str, help='Carpeta que contiene todas las imagenes')
    parser.add_argument("--captions_file", type='str', help='Archivo JSON que contiene las frases')
    parser.add_argument("--dicts", type=str, help='Diccionarios')

    args = parser.parse_args()
    main(args)

# encoder = model.Encoder(1024).cuda()
# decoder = model.Decoder(1024, 1024, 100, 10).cuda()

# input = np.random.normal(size=(32, 3, 224, 224))
# input_v = torch.tensor(input, dtype=torch.float32).cuda()

# output_v = encoder(input_v)

# phrases = decoder.sample(output_v)

# print(phrases)
