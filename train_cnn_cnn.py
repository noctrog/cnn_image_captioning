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

            # Pasar la frase a todo minusculas y separar signos de puntuacion
            captions.append([word for word in (target[random.randrange(5)]).lower()
                            .replace('.', ' .')
                            .replace(',', ' ,')
                            .replace('\'', ' \'').split(' ') if word != ''])

            #Actualizar el tensor de imagenes
            tensor_images[i] = img

        yield tensor_images, captions

## ------------------------------------------------------------
## ------------------------------------------------------------
## -------------- Bucle de entrenamiento ----------------------

def main(args):

    # Cargar fotos y captions para entrenar el modelo
    trainloader = dataloader(args.image_folder, args.captions_file, args.batch_size)

    # Cargar fotos y captions para validar el modelo
    # val_imgs = datasets.CocoCaptions(root = args.val_image_folder,
                             # annFile = args.val_captions_file,
                                # transform = transforms.Compose([
                                    # transforms.Resize(224),
                                    # transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         # std=[0.229,0.224, 0.225])]))

    # Mantiene un tensorboard con datos actualizados del entrenamiento
    writer = SummaryWriter(comment='CNN_CNN')

    # Crea los modelos a entrenar
    cnn_cnn = model.CNN_CNN_HA().to(device)

    # algoritmo que actualizara los pesos de las redes
    optimizer = optim.Adam(cnn_cnn.prediction_module.parameters(), lr=args.lr)

    # Funcion de perdida a usar: Entropia cruzada ya que los elementos a predecir (palabras)
    # son mutuamente exlusivos (solo se puede elegir una palabra)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(args.epochs):
        # Crear un generador
        trainloader = dataloader(args.image_folder, args.captions_file, args.batch_size)

        losses = []
        mean_losses = []
        for batch, (images, captions) in enumerate(trainloader):
            # Pasa las imanes a la tarjeta grafica en caso de que haya una
            images_v = images.to(device)

            # Crear embeddings para entrenar. Entradas al modelo
            train_labels = [label for label in captions]
            train_indices = [[cnn_cnn.stoi[word] if (word in cnn_cnn.stoi) else cnn_cnn.stoi['<unknown>'] for word in label] for label in train_labels]
            max_length = max([len(label) for label in train_indices])
            # Inicializa el tensor de embeddings
            train_embeddings = torch.zeros(args.batch_size, max_length+1, cnn_cnn.embedding.dim)
            # Inserta el embedding de empezar la frase (todo a 1)
            train_embeddings[:][0] = 1.0
            # train_embeddings[:][0][:] = torch.stack([first_word for _ in range(args.batch_size)], dim=0)
            # Inserta los embeddings de las palabras de cada frase
            for i, indices in enumerate(train_indices):
                train_embeddings[i][1:len(indices)+1] = cnn_cnn.embedding.vectors[indices]
            train_embeddings = train_embeddings.transpose(1, 2)     # (batch_size, embedding_size, L)
            captions_v = train_embeddings.to(device)

            # Genera la lista de indices validos para el entrenamiento, ya que muchas frases tendran
            # padding y los resultados de pasar por el padding no son validos
            valid_training_indices = []
            for i, label in enumerate(train_labels):
                valid_training_indices = valid_training_indices + [j for j in range(i*(max_length+1), i*(max_length+1) + len(label)+1)]

            # Genera el vector de salidas esperadas por el entrenamiento
            expected_ids = [[cnn_cnn.stoi[word] if word in cnn_cnn.stoi else cnn_cnn.stoi['<unknown>'] for word in label] for label in captions]
            # Anadir como palabra final el fin de clase
            for i in range(len(expected_ids)):
                expected_ids[i].append(cnn_cnn.stoi['<s>'])     # Como <s> se usa para empezar la
                # frase y nunca en la prediccion, se puede usar su posicion para predecir el final de
                # la frase
            # Desenrolla las salidas y las guarda en un tensor
            flat_expected_ids = [i for ids in expected_ids for i in ids]
            expected_v = torch.LongTensor(flat_expected_ids).to(device)

            # Genera las salidas esperadas en forma de one-hot
            # expected_labels = [label[1:] for label in captions]
            # Se usa tambien la palabra menos usada como salida en caso de que la palabra no este en el
            # diccionario (porque la entrada es embedding y la salida es one-hot, da igual)
            # expected_indices = [[cnn_cnn.stoi[word] if (word in cnn_cnn.stoi) else 399999 for word in label] for label in expected_labels]
            # Anadir el fin de linea para que el modelo pueda predecirlo (posicion 400000, no esta en el diccionario porque no hace falta)
            # expected_indices = [label + [400000] for label in expected_indices]
            # Pasar resultados esperados a un tensor one-hot
            # max_length = max([len(label) for label in expected_indices])
            # expected_v = torch.zeros((args.batch_size, max_length, cnn_cnn.vocab_size))
            # for i, indices in enumerate(expected_indices):
                # indices_v = torch.tensor(indices)
                # expected_v[i][:len(indices)] = torch.nn.functional.one_hot(indices_v, cnn_cnn.vocab_size)
            # expected_v = expected_v.to(device).long()

            # Limpia los optimizadores
            optimizer.zero_grad()

            # Calcula las predicciones
            outputs_v = cnn_cnn(images_v, captions_v)

            # Desenrolla las frases generadas para poder pasarlas por la funcion de perdida
            outputs_v = outputs_v.view(-1, cnn_cnn.vocab_size)

            # Calcula la pérdida para actualizar las redes
            loss = criterion(outputs_v[valid_training_indices], expected_v)

            # Actualizar los pesos
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Informar sobre el estado del entrenamiento, cada 100 batches
            if batch % 100 == 0:
            # if i % 100 == 0 and i != 0:
                # Actualizar tensorboard
                mean_losses.append(np.mean(losses[-100:]))
                writer.add_scalar('Training loss', mean_losses[-1])
                writer.add_text('Predicted text', 'hola')

                # Validar una foto
                # img, _ = val_imgs[random.randrange(len(val_imgs)-1)]
                # img = img.to(device)
                # img = img.view(1, *img.shape)
                # sentences = cnn_cnn.sample(img)
                # print(sentences[0])

                # Imprimir status
                print('Epoch: {}\tBatch: {}\t\tTraining loss: {}'.format(e, batch, np.mean(losses[-100:])))

                # Si la perdida esta por debajo del resto de perdidas, guardar modelo
                if mean_losses[-1] == min(mean_losses):
                    print("Modelo guardado!")
                    cnn_cnn.save()

    else:
        print('-------- Epoch: {}\t\tLoss: {} -----------'.format(e, np.mean(losses)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.002, help='Learning rate')
    parser.add_argument("--epochs", type=int, default=1, help="Numero de generaciones a entrenar")
    parser.add_argument("--batch_size", type=int, default=1, help="Tamaño del batch")
    parser.add_argument("--image_folder", type=str, help='Carpeta que contiene todas las imagenes')
    parser.add_argument("--val_image_folder", type=str, help='Carpeta que contiene todas las imagenes para validar el modelo')
    parser.add_argument("--captions_file", type=str, help='Archivo JSON que contiene las frases')
    parser.add_argument("--val_captions_file", type=str, help='Archivo JSON que contiene las frases para validar el modelo')
    parser.add_argument("--dicts", type=str, help='Diccionarios')

    args = parser.parse_args()
    main(args)
