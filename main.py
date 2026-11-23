import numpy as np
from torch import nn, optim
import torch
import math

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

def tokenEmbedding(vocab_size, embedding_size):
    tokenembedding = nn.Embedding(vocab_size, n_embd) # adi embedding table
    return tokenembedding

tokenEmbedding = tokenEmbedding(vocab_size, n_embd)



# Positional Embedding bize gosterir ki, sozler positional olaraq nece yerlesibdir
# Positional embeddingde meselen block_size = 124 bunda uje basdan baslayir deye her birinin bele deyek movqeyi tam olaraq bilinir
# Ona gore positional embeddingde block_size-dan istifade olunur.
# Her defe optimization erzinde bunlarda deyismelidir, sozun locationu hardadir onu bilmek ucun



def positionalEmbedding(block_size, embedding_size): # positional embedding
    positionalEmbedding = nn.Embedding(block_size, embedding_size)
    return positionalEmbedding

positionalEmbedding = positionalEmbedding(block_size, n_embd)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    result = e_x / e_x.sum()
    return result


def apply_causal_mask(scores):
    T = scores.size(1)
    for i in range(T):
        scores[i, i+1:] = float("-inf")   # i-ci sətirdə i+1-dən sona qədər
    return scores
# head_size ile n_head multiply olacaq ve olculeri 16 olacaq
# n_head cox olmasinin sebebi ise cumle icinde muxtelif elaqeleri oyrenmesi ile baglidir
# Sual ise niye embedding_size, head_size olmasi ile baglidir
# Embedding size olmasi -> ona gelen tokenlerinde o qeder olculur olacagi ile baglidir, multiply terefi
# Head_size olmasi ise -> axirda her biri concat olunacag multi head attentiona gore ona gore olculeri mueyyenlesdirir.


def SelfAttention(sequence_length, embedding_size,x):
    # @ -> matrix multiplication
    # softmax -> normalization etmek ucundur
    # cross-entropy -> alinmis softmaxdan -ln(softmax_result) -> loss function kimi istifade olunur
    T = sequence_length
    C = embedding_size
    sequence_length, embedding_size = x.size()
    # demeli burada x olmalidir -> x-i input kimi gotururler ve buna gorede x tam olaraq inputdur
    # burada sadece T*C formatindadir, token * channel 
    # buradada token vurulsun channel , block size ve embedding size olmalidir

    head_size = embedding_size // n_head
    # W ler - weightsdir
    # Q,k,v ise sadece bize lazim olan query,key, valuedir yeni neticeler

    Wq = torch.randn(C, head_size) # queries -> men ne axtariram
    Wk = torch.randn(C, head_size) # keys -> mene ne vere bilersen
    Wv = torch.randn(C, head_size) # value -> mene ne verdin
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    scores = ( (q @ k.T) / math.sqrt(k.size(1)))
    attention = apply_causal_mask(scores)
    attention = softmax(attention)
    attention = attention @ v

    return attention
# MultiHeadAttention hissesinde ise onlar concat olunacaq
# MultiHead Attentionda eledirki biz onu bezi hisselere ayiracayiq
# Sonra ise hemin hisseleri birlesdireceyik 
# Bundan sebeb bizde C,C seklinde olmalidir

def MultiHeadAttention(embedding_size, sequence_length,Wq, Wv, Wk):
    n_head = 8
    T = sequence_length
    C = embedding_size
    T,C = x.size()
    head_size = embedding_size // n_head # d_k
    Wq = torch.randn(C, C) # queries -> men ne axtariram
    Wk = torch.randn(C, C) # keys -> mene ne vere bilersen
    Wv = torch.randn(C, C) # value -> mene ne verdin
    
    for i in range(n_head):


        
    MultiHeadAttention = torch.cat((Wq, Wv, Wk))
    return MultiHeadAttention




def layerNorm():
    pass



def MLP():
    pass

def forward(): # x uje hazirdir indi ise sira modele vermeye gelibdr

    x = tokenEmbedding + positionalEmbedding
    x = x + Attention(layerNorm(x))
    x = x + MLP(layerNorm(x))


attention = Attention(n_embd)
print(attention)