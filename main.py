import numpy as np
from torch import nn, optim
import torch
import math
import torch.nn.functional as F


vocab_size = 1024
block_size = 128
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
device = 'cuda'
dropout = nn.Dropout(0.1)
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
    tokenembedding = nn.Embedding(vocab_size, embedding_size) # adi embedding table
    with torch.no_grad():
        tokenembedding.weight.copy_(
            torch.randn(vocab_size, embedding_size) * 0.02
        )
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
# Positinal encoding -> sinusoid ve cosinisoid kimi fix dusturlar uzerinden gede bilerik
# Yada random initialization -> random initialize edirler ve ondan sonra trainable edirler 
# ve onlari dovler erzinde training edirler

# nn.parameter -> parametrin oyrenile bilen oldugunu gosterir bizim ucun
# nn.init -> baslangic qiymetlerini mueyyenlesdirir parametrin
# standard deviation -> 0.02 dir ve bu chebiyev teoremine gore 68 faiz ehtimal demekdir mean de 0 olur


def positionalEmbedding(block_size, embedding_size): # positional embedding
    positionalEmbedding = nn.Embedding(block_size, embedding_size)
    with torch.no_grad(): # torch no grad ozumuz qesden yaziriq, gradienltlerin olacagini bildirmek ucun 
        positionalEmbedding.weight.copy_(
            torch.randn(block_size, embedding_size) * 0.02 # standard sapma
        )
    return positionalEmbedding


positionalEmbedding = positionalEmbedding(block_size, n_embd)

#endregion


# burda pos_emb ve tok_emb teyin elemek lazimdir bax bel, cunki embedding yarandi bes icine ne verirsen
# onuda bilmelisen
pos_emb = positionalEmbedding(torch.arange(sequence_length))
tok_emb = tokenEmbedding(tokens)
x = tok_emb + pos_emb
x = dropout(x)



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

# Testden bizi isteyirki qkv birlikde one linear seklinde olsun
# sadece istediyi bizden odur ki, biz birlesdirek ve sonra ayiraq
# matrix multiplication azaltmaq ucun
# model birdefelik herseyi hesablamaga calisacaq deye gpu utilization yuksek olacaq


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
    Wqkv = nn.Linear(C, 3*C, bias=False)

    qkv = Wqkv(x)
    q,k,v = qkv.split(C, dim = -1)
    
    # uje q,k,v nin olcusu bizde alinir sequence length * head_size olur
    scores = ( (q @ k.T) / math.sqrt(k.size(1)))
    attention = apply_causal_mask(scores)
    attention = nn.softmax(attention,dim = -1) # dimensionaltiy rowlar uzerinde aparilmasini isteyirik
    # her bir rowun normalizasiyasi ele olsun ki cemleri 1 e beraber olsun 

    attention = attention @ v

    return attention

# endregion


# MultiHeadAttention hissesinde ise onlar concat olunacaq
# MultiHead Attentionda eledirki biz onu bezi hisselere ayiracayiq
# Sonra ise hemin hisseleri birlesdireceyik 
# Bundan sebeb bizde C,C seklinde olmalidir
# Funksiyanin icinde yazmaq serf elemeyecek, cunki bizde her loop erzinde, funksiya isleyecek
# Burda problem yoxdur, amma problem odur ki, o andan etibaren herdefe yenisi yaradilir, kohnesi cumulative olmur
# Bu sebebi aradan qaldirmaq ucun initialization muhitini kenarda etmek lazimdir
def MultiHeadAttention(embedding_size, sequence_length,Wqkv,x, Wo,n_head = 4):
    T = sequence_length
    C = embedding_size
    T,C = x.size()
    head_size = embedding_size // n_head # d_k

    heads = []
    qkv = Wqkv(x)
    # Burdada c ile olunur, amma ki eger q heads, k heads ve artiq v heads tapanda onda uje avtomatik olaraq heads size kecirik
    q,k,v = qkv.split(C, dim = -1)    
    q_heads = q.split(head_size, dim=-1)
    k_heads = k.split(head_size, dim=-1)
    v_heads = v.split(head_size, dim=-1)
    for i in range(n_head):
        q = q_heads[i]
        v = v_heads[i]
        k = k_heads[i]
        scores = ( (q @ k.T) / math.sqrt(k.size(1)))
        attention = apply_causal_mask(scores)
        attention = softmax(attention, dim = -1)
        attention = dropout(attention)
        attention = attention @ v
        heads.append(attention)


    result = torch.cat(heads, dim=-1)  # Concatenate along the last dimension
    # burda bize birdenede elave olaraq lazimdirki headler duz elaqededir
    # amma Wo olarsa onda uje elave olaraq hemcinin bizde diger headler ile elaqede qurulur
    result = Wo(result) # eger zaten Linear function kimi yaradirsansa bir novu onun icine alirsan
    # son netice olaraq bilirsen ne cixir linear function seklinde: (T,128) @ (128,128) = (T, 128)
    return result


Wqkv = nn.Linear(n_embd, 3*n_embd, bias = False)
nn.init.normal_(Wqkv.weight, mean = 0.0, std = 0.02)
Wo = nn.Linear(n_embd, n_embd, bias = False) # bunu ona gore yaziriq ki butun headler bir biri ile danisa bilsin
nn.init.normal_(Wo.weight, mean = 0.0, std = 0.02)


# gamma ve beta trainable parametrlerdir
# normasllasdirmadan sonra mean = 0 
# variance = 1
# gamma miqyasi deyisir meselen bize 1 ve -1 arasi yoxda 5 min ve -5 min arasi laizmdir
# beta ise onlari shift edir 
# gamma hemcinin bizim dispersiya yayilmasimiza boyudur
# beta ise biz daha cox eded oxu uzerinde menfi veya musbet olan hisselerde ehreketimize baxiriq


def layerNorm(sequence_length, embedding_size, gamma,x, beta, epslon = 3e-4):
    T = sequence_length
    C = embedding_size
    T,C = x.size()
    mean = x.mean(dim=-1, keepdim=True) # dimensionalityni saxlamaq ucun
    variance = ((x - mean) ** 2 ).mean(dim = -1, keepdim = True)

    y = (gamma * ( (x - mean) /  ( (variance + epslon) ** (0.5)) ) + beta)
    return y


gamma = nn.Parameter(torch.ones(n_embd)) # scale etmek ucundur
beta = nn.Parameter(torch.zeros(n_embd)) #shift etmek ucundur

# epslon sadece bu asagidaki hisse menfi falan olmasin deye, stabilliyi qoruyur o qeder


# residual connection -> evvelki melumatlari unutmamaq ucun istifade olunan mexanizmdir
# residual connection bize melumatlari multi head attention terefden getirir
# ele bil attention mexanizmini unutmut

def add(x, output):
    return x + output


# Multi Layer perception
# bu hisse hemsinin position-wise feed-forward network adlanir





def MLP(sequence_length, embedding_size,x, W1, W2):
    hidden_size = embedding_size * 4


    x = x @ W1 + b1 # artiq daha boyuk olculu shape icine girdi
    x = F.gelu(x)
    x = x @ W2 + b2

    return x
    

W1 = nn.Linear(n_embd, 4*n_embd, bias=False)
W2 = nn.Linear(4*n_embd, n_embd, bias=False)

nn.init.normal_(W1.weight, mean=0.0, std=0.02)
nn.init.normal_(W2.weight, mean=0.0, std=0.02)

def forward(): # x uje hazirdir indi ise sira modele vermeye gelibdr

    x = tokenEmbedding + positionalEmbedding
    x = x + Attention(layerNorm(x))
    x = x + MLP(layerNorm(x))

dusunurem ki pis yazmamisam