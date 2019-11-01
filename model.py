import os
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from torchvision import models
# from torchtext import vocab
from mini_glove import MiniGlove

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
    def __init__(self, pretrained=False):
        super(VisionModule, self).__init__()

        if not pretrained and os.path.exists('./weights/vgg.dat'):
            state_dict = torch.load('./weights/vgg.dat')
            self.convnet = models.vgg16(pretrained=pretrained).features[:-1].load_state_dict(state_dict)
        else:
            print('Usando el modelo vgg preentrenado')
            self.convnet = models.vgg16(pretrained=True).features[:-1]

    def forward(self, x):
        # Pasar la imagen por la red convolucional
        # Entrada: imagen de (batch, 3, 224, 224)
        # Salida:            (batch, 512, 14, 14), por lo que d = 14 y Dc = 512
        return self.convnet(x)

    def save(self):
        state_dict = self.convnet.state_dict()
        torch.save('./weights/vgg.dat', state_dict)

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

# Convolucion causal (solo tiene en cuenta palabras anteriores)
# class CausalConv1d(nn.Module):
    # # kernel_size, tamaño del kernel causal, el kernel interno sera simétrico pero con los números
    # # de la izquierda a 0
    # def __init__(self, kernel_size, embedding_dim):
        # super(CausalConv1d, self).__init__()

        # self.embedding_dim = embedding_dim
        # self.k = kernel_size
        # self.causal_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim,
                                     # kernel_size=(2 * self.k - 1), padding=self.k-1)

    # def forward(self, x):
        # # x: (batch, embedding_dim, length)

        # # Poner los valores que esten a la derecha del kernel a 0
        # # TODO: buscar solucion más elegante para hacer una convolucion causal
        # with torch.no_grad():
            # self.causal_conv.weight[:, :, self.k:] = 0
            # self.causal_conv.bias[self.k:] = 0

        # return self.causal_conv(x)

# Capa de convolucion (con Gated Linear Unit de activacion)
class CausalConvolutionLayer(nn.Module):
    def __init__(self, kernel_size, embedding_dim):
        super(CausalConvolutionLayer, self).__init__()

        self.convolution_a = CausalConv1d(embedding_dim, embedding_dim, kernel_size)
        self.convolution_b = CausalConv1d(embedding_dim, embedding_dim, kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, embedding_dim, L)
        h_a = self.convolution_a(x)
        h_b = self.convolution_b(x)
        return h_a * self.sigmoid(h_b)

# Modulo de lenguaje
class LanguageModule(nn.Module):
    # Modulo de lenguaje sin atencion jerarquica
    def __init__(self, embedding_dim=300, n_layers=6, kernel_size=3):
        super(LanguageModule, self).__init__()

        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim

        conv_list = [CausalConvolutionLayer(self.kernel_size, self.embedding_dim)
                     for i in range(self.n_layers)]

        self.convolutions = nn.Sequential(*conv_list)

    def forward(self, x):
        # x:    (batch, embedding_size, L)
        return self.convolutions(x)

# Capa de atencion
class AttentionModule(nn.Module):
    def __init__(self, image_vectors, embedding_dim):
        super(AttentionModule, self).__init__()

        self.De = embedding_dim
        self.Dc = image_vectors
        self.U = torch.nn.Parameter(data=torch.Tensor(self.De, self.Dc), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, c, v):
        v = v.view(v.shape[0], v.shape[1], -1)
        # c: (batch, embedding_dim, L)
        # v: (batch, image_features, N)
        aux = torch.matmul(c.transpose(1, 2), self.U)   # aux: (batch, L, image_features)
        S = torch.matmul(aux, v)                        # S:   (batch, L, N)
        w = self.softmax(S)                             # w:   (batch, L, N)
        a = torch.matmul(w, v.transpose(1, 2))        # aux: (batch, L, Dc)
        return a.transpose(1, 2)

# Modulo de prediccion
class PredictionModule(nn.Module):
    def __init__(self, image_vectors, embedding_dim, vocab_size, hidden_layer=1024):
        super(PredictionModule, self).__init__()

        self.Dc = image_vectors
        self.De = embedding_dim
        self.hidden_layer = hidden_layer
        self.vocab_size = vocab_size

        self.convolution_a = nn.Conv1d(image_vectors, self.hidden_layer, 1, bias=True)
        self.convolution_c = nn.Conv1d(embedding_dim, self.hidden_layer, 1, bias=True)
        self.linear = nn.Linear(self.hidden_layer, self.vocab_size, bias=False)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, a, c):
        # a:    (batch, Dc, L)
        # c:    (batch, De, L) 
        h = self.leakyrelu(self.convolution_a(a) + self.convolution_c(c))   # h:    (batch, hidden_size, L)

        # batch_size = h.shape[0]
        # L = h.shape[-1]
        # h = h.reshape(h.shape[0], -1)
        h = h.transpose(1, 2)       # h:    (batch, L, hidden_size)
        P = self.linear(h)          # P:    (batch, L, vocab_size)
        return self.log_softmax(P)      # P:    (batch, L, vocab_size)


# Modelo de lenguaje
class CNN_CNN(nn.Module):
    # Modelo completo de CNN + CNN sin atencion jerarquica
    def __init__(self, max_length=20, embedding=None, train_cnn=False):
        super(CNN_CNN, self).__init__()

        ## Parametros
        # k: tamaño del kernel
        self.k = 3
        # Longitud maxima de las frases generadas
        self.max_length = max_length

        # Embedding: si no se especifica ninguno, usar GloVe 
        if embedding == None:
            self.embedding = MiniGlove()
            self.embedding.load()
            if use_cuda:
                self.embedding.vectors = self.embedding.vectors.cuda()
            self.vocab_size = self.embedding.vectors.shape[0]
        else:
            self.embedding = embedding
            self.vocab_size = embedding.num_embeddings

        # Modulo de vision
        self.vision_module = VisionModule()
        # Fijar los pesos de la red convolucional para las imagenes
        if not train_cnn:
            for param in self.vision_module.parameters():
                param.requires_grad = False

        # Modulo de lenguaje
        self.language_module = LanguageModule()

        # Modulo de atencion
        self.attention_module = AttentionModule(512, 300)

        # Modulo de prediccion
        self.prediction_module = PredictionModule(512, 300, self.vocab_size)

    def forward(self, img, caption):
        # img:  (batch, channels, 224, 224)
        # caption: (batch, embed_size, L)
        v = self.vision_module(img)
        c = self.language_module(caption)
        a = self.attention_module(c, v)
        P = self.prediction_module(a, c)
        return P

    def sample(self, img):
        # Crear un caption solo con el start
        # caption = self.embedding.vectors[self.embedding.stoi['<S>']]
        caption = self.embedding.vectors[self.embedding.stoi['super']]
        caption = caption.reshape((1, self.embedding.dim, 1))
        caption = caption.repeat(img.shape[0], 1, 1)

        sentences = [list() for _ in range(caption.shape[0])]

        # Generar palabras hasta obtener </S> o llegar a la longitud maxima
        for _ in range(self.max_length):
            prediction = self.forward(img, caption)                 # prediction:       (batch, L, vocab_size)
            _, word_ids = torch.max(prediction[:, -1, :], dim=1)    # word_ids:         (batch)
            word_ids = word_ids.to(torch.int32)
            new_words = self.embedding.vectors[word_ids.cpu().numpy()]
            new_words = new_words.view(new_words.shape[0], 1, new_words.shape[1])                 # new_words:        (batch, 1, embedding_size)
            caption = torch.cat([caption, new_words.transpose(1, 2)], dim=2)

            # append words
            for i, sentence in enumerate(sentences):
                sentence.append(self.embedding.itos[word_ids.cpu().numpy()[i]])
            # if word_id == self.embedding.stoi['</S>']:
                # break

        # Generar frases
        for sentence in sentences:
            sentence = " ".join(sentence)

        return sentences

    def save(self):
        torch.save(self.state_dict(), './weights/cnn_cnn.dat')
        pass

class HierarchicalAttentionLayer(nn.Module):
    """Modulo de atencion jerarquico"""
    def __init__(self, kernel_size=3, embedding_dim=300, image_features=512):
        super(HierarchicalAttentionLayer, self).__init__()

        self.convolution_a = CausalConv1d(embedding_dim, embedding_dim, kernel_size)
        self.convolution_b = CausalConv1d(embedding_dim, embedding_dim, kernel_size)
        self.sigmoid = nn.Sigmoid()

        self.Wa = nn.Linear(image_features, embedding_dim)
        self.Wb = nn.Linear(image_features, embedding_dim)

        self.attention_module = AttentionModule(image_features, embedding_dim)

    def forward(self, h, v):
        # x: (batch, embedding_dim, L)
        a = self.attention_module(h, v)                     # a:    (batch, image_features, L)
        a_a = self.Wa(a.transpose(1, 2)).transpose(1, 2)    # a_a:  (batch, embedding_dim, L)
        a_b = self.Wb(a.transpose(1, 2)).transpose(1, 2)    # a_b:  (batch, embedding_dim, L)
        h_a = self.convolution_a(h) + a_a
        h_b = self.convolution_b(h) + a_b
        return h_a * self.sigmoid(h_b), a

class HierarchicalAttentionModule(nn.Module):
    def __init__(self, n_layers=6, kernel_size=3, embedding_dim=300, image_features=512):
        super(HierarchicalAttentionModule, self).__init__()

        self.attention_layers = nn.ModuleList([HierarchicalAttentionLayer(kernel_size=kernel_size,
                                                                          embedding_dim=embedding_dim,
                                                                          image_features=image_features)
                                               for _ in range(n_layers)])
    def forward(self, h, v):
        for i in range(len(self.attention_layers) - 1):
            h, _ = self.attention_layers[i](h, v)

        h, a = self.attention_layers[-1](h, v)
        return h, a

class CNN_CNN_HA(nn.Module):
    """CNN_CNN con modulo de atencion jerarquico"""
    def __init__(self, n_layers=6, max_length=15, embedding=None, train_cnn=False):
        super(CNN_CNN_HA, self).__init__()

        ## Parametros
        # k: tamaño del kernel
        self.k = 3
        # Longitud maxima de las frases generadas
        self.max_length = max_length

        # Embedding: si no se especifica ninguno, usar GloVe 
        if embedding == None:
            self.embedding = MiniGlove()
            self.embedding.load()
            if use_cuda:
                self.embedding.vectors = self.embedding.vectors.cuda()
            self.vocab_size = self.embedding.vectors.shape[0]
            self.stoi = self.embedding.stoi
            self.itos = self.embedding.itos
        else:
            self.embedding = embedding
            self.vocab_size = embedding.num_embeddings

        # Modulo de vision
        self.vision_module = VisionModule()
        # Fijar los pesos de la red convolucional para las imagenes
        if not train_cnn:
            for param in self.vision_module.parameters():
                param.requires_grad = False

        # Modulo de lenguaje-atencion
        self.language_module_att = HierarchicalAttentionModule()

        # Modulo de prediccion
        self.prediction_module = PredictionModule(512, 300, self.vocab_size)

    def forward(self, img, caption):
        v = self.vision_module(img)
        c, a = self.language_module_att(caption, v)
        P = self.prediction_module(a, c)
        return P

    def sample(self, img):
        # Crear un caption solo con el start
        # En caso de usar GloVe, como no tiene una palabra para indicar el inicio de la frase, se
        # escoge la palabra con menos frecuencia, que a su vez sera la mas alejada del centro y por
        # tanto se podra destinguir facilmente con el resto de palabras
        start_word = '<s>'
        caption = self.embedding.vectors[self.embedding.stoi[start_word]]
        caption = caption.reshape((1, self.embedding.dim, 1))
        caption = caption.repeat(img.shape[0], 1, 1)

        sentences = [list() for _ in range(caption.shape[0])]

        # Generar palabras hasta obtener </S> o llegar a la longitud maxima
        for _ in range(self.max_length):
            log_prediction = self.forward(img, caption)                 # prediction:       (batch, L, vocab_size)
            prediction = torch.exp(log_prediction)
            _, word_ids = torch.max(prediction[:, -1, :], dim=1)    # word_ids:         (batch)
            word_ids = word_ids.to(torch.int32)
            new_words = self.embedding.vectors[word_ids.cpu().numpy()]
            new_words = new_words.view(new_words.shape[0], 1, new_words.shape[1])                 # new_words:        (batch, 1, embedding_size)
            caption = torch.cat([caption, new_words.transpose(1, 2)], dim=2)

            # append words
            for i, sentence in enumerate(sentences):
                sentence.append(self.embedding.itos[word_ids.cpu().numpy()[i]])

        # Generar frases
        for sentence in sentences:
            sentence = " ".join(sentence)

        return sentences

    def save(self):
        torch.save(self.state_dict(), './weights/cnn_cnn_ha.dat')

    def load(self):
        if os.path.exists('./weights/cnn_cnn_ha.dat'):
            self.load_state_dict(torch.load('./weights/cnn_cnn_ha.dat'))
            print('Modelo cargado correctamente')
