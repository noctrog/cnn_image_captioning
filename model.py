import os
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from torchvision import models
from torchtext import vocab

# Comprueba que haya una GPU compatible con CUDA
use_cuda = torch.cuda.is_available()

if use_cuda:
    print('Usando la GPU')
else:
    print('Usando la CPU')


class Encoder(nn.Module):
    def __init__(self, latent_size=1024):
        super(Encoder, self).__init__()

        # Parametros del modelo
        self.latent_size = latent_size
        self.pretrained = not os.path.exists('./weights/encoder.dat')

        # Cargar la red convolucional
        self.cnn = models.resnet50(pretrained=self.pretrained)

        # Cambiar el clasificador por una red densa que genere un punto en el espacio lantente
        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000, self.latent_size)
        )

    def forward(self, x):
        # Pasar batch a traves de la CNN y obtener espacios latentes
        x = self.cnn(x)
        x = self.classifier(x)
        return x

    def save(self):
        dump = {'latent_size': self.latent_size,
                'state_dict': self.state_dict()}
        if (not os.path.exists('./weights')):
            os.mkdir('./weights')
        torch.save(dump, './weights/encoder.dat')

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, vocab_size, max_seq_length, dropout=0.2, n_layers=2):
        super(Decoder, self).__init__()

        # Parametros del modelo
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Embedding: para pasar de palabras a posiciones en un espacio hiperdimensional
        self.embedding = nn.Embedding(vocab_size, latent_size)

        # Red recurrente
        self.lstm = nn.LSTM(input_size=latent_size, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True, dropout=dropout)

        # Capa FC para mapear salidas del LSTM
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        """Decodifica del espacio latente y genera los captions"""
        # Resetea los estados internos de la red recurrente
        hidden = self.init_hidden(features.shape[0])

        # Pasar las palabras por el embedding
        embeddings = self.embed(captions)

        # Insertar la representacion de la foto en el primer elemento de cada frase
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        # De esta forma los LSTM pararan de leer cuando encuentren el primer 0 (0 padding)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        # Obtener las salidas de la red recurrente
        hiddens, _ = self.lstm(packed, hidden)

        # Desenrolla todos los hidden
        outputs = self.dropout(hiddens).contiguous().view(-1, self.hidden_size)

        # Calcular predicciones de las siguientes palabras
        outputs = self.linear(outputs)   # TODO: [0] por que???

        # Devuelve outputs a su forma original
        outputs = outputs.view(hiddens.shape)

        # Devolver resultados
        return outputs

    def sample(self, features):
        """EJECUTAR EN MODO EVAL. Generar captions a partir de imagenes usando greedy search"""
        # TODO: add temperature

        sampled_ids = []
        current_input = features.unsqueeze(1)

        # Resetea el estado interno de la red recurrente
        states = self.init_hidden(features.shape[0])    # hidden: (batch_size, 1, hidde_size)

        for _ in range(self.max_seq_length):
            # Avanza un caracter por la LSTM
            hiddens, states = self.lstm(current_input, states)  # hiddens: (batch_size, 1, hidden_size)

            # Calcular las predicciones de los id de las siguientes palabras
            output = self.linear(hiddens.squeeze(1))            # output: (batch_size, vocab_size) 

            # Elegir la prediccion con mayor probabilidad
            _, predicted = output.max(1)                        # predicted: (batch_size)

            # Guardar predicciones
            sampled_ids.append(predicted)

            # Calcular punto en el espacio latente de las palabras escogidas
            current_input = self.embedding(predicted)           # current_input: (batch_size, latent_size)

            # Añadir la dimension para la secuencia (solo un caracter), para poder ser leida por la LSTM
            current_input = current_input.unsqueeze(1)

        # Convertir la lista de tokens en un tensor
        sampled_ids = torch.stack(sampled_ids, 1)               # sampled_ids: (batch_size, max_seq_length)

        # Devolver valores obtenidos
        return sampled_ids

    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data

        if (use_cuda):
            hidden = (weights.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weights.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weights.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weights.new(self.n_layers, batch_size, self.hidden_size).zero_())

        return hidden

    def save(self):
        dump = {'state_dict': self.state_dict(),
                'latent_size': self.latent_size,
                'hidden_size': self.hidden_size,
                'vocab_size': self.vocab_size,
                'n_layers': self.n_layers,
                'dropout': self.dropout_p}
        if (not os.path.exists('./weights')):
            os.mkdir('./weights')
        torch.save(dump, './weights/decoder.dat')


## ---------------- CNN - CNN ------------------
# Basado en el paper: CNN+CNN: Convoutional decoders for Image Captioning

# Modulo de vision
class VisionModule(nn.Module):
    def __init__(self):
        super(VisionModule, self).__init__()
        # Cargar modelo preentrenado VGG, deshacerse del clasificador y del ultimo max pool
        pretrained = True if not os.path.exists('./weights/vgg.dat') else False
        self.convnet = models.vgg16(pretrained=pretrained).features[:-1]

        # si se ha encontrado el archivo con los pesos guardados, cargarlo
        if not pretrained:
            state_dict = torch.load('./weights/vgg.dat')
            self.convnet.load_state_dict(state_dict)

    def forward(self, x):
        # Pasar la imagen por la red convolucional
        # Entrada: imagen de (batch, 3, 224, 224)
        # Salida:            (batch, 512, 14, 14), por lo que d = 14 y Dc = 512
        return self.convnet(x)

    def save(self):
        state_dict = self.convnet.state_dict()
        torch.save('./weights/vgg.dat', state_dict)

# Convolucion causal (solo tiene en cuenta palabras anteriores)
class CausalConv1d(nn.Module):
    # kernel_size, tamaño del kernel causal, el kernel interno sera simétrico pero con los números
    # de la izquierda a 0
    def __init__(self, kernel_size, embedding_dim):
        super(CausalConv1d, self).__init__()

        self.embedding_dim = embedding_dim
        self.k = kernel_size
        self.causal_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim,
                                     kernel_size=(2 * self.k - 1), padding=self.k-1)

    def forward(self, x):
        # x: (batch, embedding_dim, length)

        # Poner los valores que esten a la derecha del kernel a 0
        # TODO: buscar solucion más elegante para hacer una convolucion causal
        with torch.no_grad():
            self.causal_conv.weight[:, :, self.k:] = 0

        return self.causal_conv(x)

# Capa de convolucion (con Gated Linear Unit de activacion)
class CausalConvolutionLayer(nn.Module):
    def __init__(self, kernel_size, embedding_dim):
        super(CausalConvolutionLayer, self).__init__()

        self.convolution_a = CausalConv1d(kernel_size, embedding_dim)
        self.convolution_b = CausalConv1d(kernel_size, embedding_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_a = self.convolution_a(x)
        h_b = self.convolution_b(x)
        return h_a * self.sigmoid(h_b)

# Modelo de lenguaje
class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()

        ## Parametros
        # k: tamaño del kernel
        self.k = 3

        # Embedding (de momento usar GloVe preentrenado)
        self.embedding = vocab.GloVe(name='6B', dim=300)

        # 



