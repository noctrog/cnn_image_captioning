import random
import argparse
import os.path as path

import numpy as np

import torch
from torch import optim
from torchvision import datasets, transforms

import model

from tensorboardX import SummaryWriter

# Usar CUDA si esta disponible
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

## ------------------------------------------------------------
## ------------------------------------------------------------
## -------------- Leer los datos y procesarlos ----------------

# Las imagenes tienen que ir de 0 a 1, con un tamanio de 224x224
# despues normalizadas con normalize =
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,
# 0.224, 0.225]) porque la red convolucional se entreno con
# imagenes con este formato

def dataloader(image_folder, captions_file, batch_size):

    #Definir el tensor para guardar las imagenes de un batch
    tensor_images = torch.zeros((batch_size, 3, 224, 224)).to(device)
    #Definir las transformaciones que se aplican a las imagenes
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])])

    #Cargar el dataset
    cap = datasets.CocoCaptions(root = image_folder,
                             annFile = captions_file,
                             transform = transform)

    for batch in range(int( len(cap)/batch_size )):
        captions = []
        for i in range(batch_size):
            #Obtener una imagen con sus captions y seleccionar uno al azar
            img, target = cap[batch*batch_size+i]
            captions.append(target[random.randrange(5)])

            #Actualizar el tensor de imagenes
            tensor_images[i] = img

        yield tensor_images, captions

## ------------------------------------------------------------
## ------------------------------------------------------------
## -------------- Bucle de entrenamiento ----------------------

def main(args):

    # Crear un generador
    trainloader = dataloader(args.image_folder, args.captions_file, args.batch_size)
    img, labels = next(trainloader)
    print(img)
    print(labels)
    return

    # Mantiene un tensorboard con datos actualizados del entrenamiento
    writer = SummaryWriter(comment='CNN_CNN')

    # Crea los modelos a entrenar
    cnn_cnn = model.CNN_CNN().to(device)

    # algoritmo que actualizara los pesos de las redes
    optimizer = optim.AdamW(cnn_cnn.parameters(), lr=args.lr)

    # Funcion de perdida a usar: Entropia cruzada ya que los elementos a predecir (palabras)
    # son mutuamente exlusivos (solo se puede elegir una palabra)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(args.epochs):
        # TODO: generador

        losses = []
        for i, images, captions in enumerate(trainloader):
            # Lee el siguiente batch
            images_v = images.to(device)
            captions_v = captions.to(device)

            # TODO: Genera las salidas esperadas en forma de one-hot
            # expected_v =

            # Limpia los optimizadores
            optimizer.zero_grad()

            # Calcula las predicciones
            outputs_v = cnn_cnn(images_v, captions_v)

            # Calcula la pérdida para actualizar las redes
            loss = criterion(outputs_v, expected_v)

            # Actualizar los pesos
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
    parser.add_argument("--captions_file", type=str, help='Archivo JSON que contiene las frases')
    parser.add_argument("--dicts", type=str, help='Diccionarios')

    args = parser.parse_args()
    main(args)
