import numpy as np
from torch import nn, optim
import torch
import math
from torch import nn as F


vocab_size = 1024
block_size = 128
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
device = 'cuda'
batch_size = 4 # her gelen data 4 hisseye ayrilib ele verilecekdir.( mini batches mentiqi)

# Token embedding and Positional embedding
"""
Pre Layer Norm -> x = x + Attn(Ln1(x))
after x + MLP(LN2(x))





"""


# token embedding trainable parameterse aiddir
# dataset text -> encoding by bpe algorithm -> token embedding


# region Token Embedding
def tokenEmbedding(vocab_size, embedding_size):
    tokenembedding = nn.Embedding(vocab_size, n_embd) # adi embedding table
    return tokenembedding

tokenEmbedding = tokenEmbedding(vocab_size, n_embd)

# endregion


# region Positional Embedding
# Positional Embedding bize gosterir ki, sozler positional olaraq nece yerlesibdir
# Positional encodingde esas odur ki, bizim ucun sequence lengthdeki tokenler
# Yeni meselen salam necesen ile salam neyniyirsen bir birinden cox ferqlenir
# ona gore her bir cut yerde duranlar ucun ayri bir dustur
# tek yerde duranlar ucun ayri bir dustur
# block size ile eliyendede sequence length hissesine qeder gozlemeliyik



def positionalEmbedding(block_size, embedding_size): # positional embedding
    positionalEmbedding = nn.Embedding(block_size, embedding_size)
    position = torch.arange(0, block_size, dtype=torch.long).unsqueeze(1) #column uzerinden getmeliyik
    div_term = torch.exp(
        torch.arange(0, embedding_size, 2) * 
        (-math.log(10000.0) / embedding_size)
    )

    positionalEmbedding[:, 0::2] = torch.sin(position * div_term)
    positionalEmbedding[:, 1::2] = torch.cos(position * div_term)

    return positionalEmbedding


positionalEmbedding = positionalEmbedding(block_size, n_embd)

#endregion

x = tokenEmbedding + positionalEmbedding[:sequence_length]


# softmax function bizim ucun butun tokenler arasinda ehtimal paylanmasi etmek ucundur
# tokenler arasinda hansinin daha cox ehtimal olacagini gosterir
# p / p.sum() -> normalization ucundur
# yeni bu tokenlerin logitsini alib softmaxdan keciririk ve bu tokenlerin ehtimalini tapiriq

# region
def softmax(x):
    e_x = np.exp(x - np.max(x))
    result = e_x / e_x.sum()
    return result

# endregion

# causal mask geleceyi gormemek, ancag kecmis ile elaqede qalmaq ucun istifade olunur
# burda ise range funksiyasi ile yazmisiq.

# region
def apply_causal_mask(scores):
    T = scores.size(1)
    for i in range(T):
        scores[i, i+1:] = float("-inf")   # i-ci sətirdə i+1-dən sona qədər
    return scores
# endregion
# head_size ile n_head multiply olacaq ve olculeri 16 olacaq
# n_head cox olmasinin sebebi ise cumle icinde muxtelif elaqeleri oyrenmesi ile baglidir
# Sual ise niye embedding_size, head_size olmasi ile baglidir
# Embedding size olmasi -> ona gelen tokenlerinde o qeder olculur olacagi ile baglidir, multiply terefi
# Head_size olmasi ise -> axirda her biri concat olunacag multi head attentiona gore ona gore olculeri mueyyenlesdirir.




# Adi self attentionda head_size qurmagina bele ehtoyyac yoxdur!

# region Self Attention
def SelfAttention(sequence_length, embedding_size,x):
    # @ -> matrix multiplication
    # softmax -> normalization etmek ucundur
    # cross-entropy -> alinmis softmaxdan -ln(softmax_result) -> loss function kimi istifade olunur
    T = sequence_length
    C = embedding_size
    # bu cumlenin uzunlugu ve ona aid embeddingler ile yeni positional embedding
    sequence_length, embedding_size = x.size()
    # demeli burada x olmalidir -> x-i input kimi gotururler ve buna gorede x tam olaraq inputdur
    # burada sadece T*C formatindadir, token * channel 
    # buradada token vurulsun channel , block size ve embedding size olmalidir

    head_size = embedding_size // n_head 
    # W ler - weightsdir
    # Q,k,v ise sadece bize lazim olan query,key, valuedir yeni neticeler

    # torch randomizasiya ederek bizim ucun lazim olan weightleri verir

    Wq = torch.randn(C, head_size) # queries -> men ne axtariram
    Wk = torch.randn(C, head_size) # keys -> mene ne vere bilersen
    Wv = torch.randn(C, head_size) # value -> mene ne verdin
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    # uje q,k,v nin olcusu bizde alinir sequence length * head_size olur
    scores = ( (q @ k.T) / math.sqrt(k.size(1)))
    attention = apply_causal_mask(scores)
    attention = softmax(attention)
    attention = attention @ v

    return attention

# endregion


# MultiHeadAttention hissesinde ise onlar concat olunacaq
# MultiHead Attentionda eledirki biz onu bezi hisselere ayiracayiq
# Sonra ise hemin hisseleri birlesdireceyik 
# Bundan sebeb bizde C,C seklinde olmalidir

def MultiHeadAttention(embedding_size, sequence_length,Wq, Wv, Wk,x):
    n_head = 8
    T = sequence_length
    C = embedding_size
    T,C = x.size()
    head_size = embedding_size // n_head # d_k

    heads = []
    
    for i in range(n_head):
        Wq = torch.randn(C, head_size) # queries -> men ne axtariram
        Wk = torch.randn(C, head_size) # keys -> mene ne vere bilersen
        Wv = torch.randn(C, head_size) # value -> mene ne verdin

        q = x @ Wq
        k = x @ Wk
        v = x @ Wv

        scores = ( (q @ k.T) / math.sqrt(k.size(1)))
        attention = apply_causal_mask(scores)
        attention = softmax(attention, dim = -1)
        attention = attention @ v
        heads.append(attention)


    result = torch.cat(heads, dim=-1)  # Concatenate along the last dimension
    # burda bize birdenede elave olaraq lazimdirki headler duz elaqededir
    # amma Wo olarsa onda uje elave olaraq hemcinin bizde diger headler ile elaqede qurulur
    Wo = torch.randn(C,C)
    result = result * Wo
    return result



# gamma ve beta trainable parametrlerdir
# normasllasdirmadan sonra mean = 0 
# variance = 1
# gamma miqyasi deyisir meselen bize 1 ve -1 arasi yoxda 5 min ve -5 min arasi laizmdir
# beta ise onlari shift edir 
# gamma hemcinin bizim dispersiya yayilmasimiza boyudur
# beta ise biz daha cox eded oxu uzerinde menfi veya musbet olan hisselerde ehreketimize baxiriq


def layerNorm(sequence_length, embedding_size, epslon = 0.02):
    T = sequence_length
    C = embedding_size
    T,C = x.size()
    mean = x.mean(dim=-1, keepdim=True) # dimensionalityni saxlamaq ucun
    variance = ((x - mean) ** 2 ).mean(dim = -1, keepdim = True)
    gamma = nn.Parameter(torch.ones(embedding_size))
    beta = nn.Parameter(torch.zeros(embedding_size))

    y = (gamma * ( (x - mean) /  ( (variance + beta) ** (-0.5)) )   + beta)
    return y


# residual connection -> evvelki melumatlari unutmamaq ucun istifade olunan mexanizmdir
# residual connection bize melumatlari multi head attention terefden getirir
# ele bil attention mexanizmini unutmut

def add(x, output):
    return x + output


# Multi Layer perception
# bu hisse hemsinin position-wise feed-forward network adlanir





def MLP(sequence_length, embedding_size,x):
    hidden_size = embedding_size * 4
    W1 = nn.Parameter(torch.randn(embedding_size,hidden_size) * 0.02) # bu initiazalition ucundur ki, cox boyuk baslamasin
    W2 = nn.Parameter(torch.randn(hidden_size, embedding_size) * 0.02)

    b1 = nn.Parameter(torch.zeros(hidden_size))
    b2 = nn.Parameter(torch.zeros(embedding_size))


    x = x @ W1 + b1 # artiq daha boyuk olculu shape icine girdi
    x = F.gelu(x)
    x = x @ W2 + b2

    return x
    

def forward(): # x uje hazirdir indi ise sira modele vermeye gelibdr

    x = tokenEmbedding + positionalEmbedding
    x = x + Attention(layerNorm(x))
    x = x + MLP(layerNorm(x))

dusunurem ki pis yazmamisam