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
    optimizer = optim.AdamW(parameters, lr=args.lr)

    # Funcion de perdida a usar: Entropia cruzada ya que los elementos a predecir (palabras)
    # son mutuamente exlusivos (solo se puede elegir una palabra)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(args.epochs):
        # Lee las imagenes con una frase distinta cada vez
        trainloader = dataloader(args.image_folder, args.captions_file, args.batch_size)

        for images, captions, lengths in trainloader:
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

            # Calcula la pérdida para actualizar las redes
            loss = criterion(outputs, captions_v)

            # Actuaslizar los pesos 
            loss.backward()
            optimizer.step()

    else:
        print('Epoch {}:', e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.002, help='Learning rate')
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--image_folder", type=str, help='Where training images are located')
    parser.add_argument("--captions_file", type='str', help='JSON containing captions for each image')

    args = parser.parse_args()
    main(args)

# encoder = model.Encoder(1024).cuda()
# decoder = model.Decoder(1024, 1024, 100, 10).cuda()

# input = np.random.normal(size=(32, 3, 224, 224))
# input_v = torch.tensor(input, dtype=torch.float32).cuda()

# output_v = encoder(input_v)

# phrases = decoder.sample(output_v)

# print(phrases)
