# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 06:24:16 2022

@author: prana
"""

import matplotlib.pyplot as plt
import numpy as np
import re
sentences = """We are about to study the idea of a computational 
process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called 
data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In 
effect,
we conjure the spirits of the computer with our spells."""
# sentences = re.sub('[^A-Za-z0-9]+', ' ', sentences)
# sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip()
sentences = sentences.lower()
words = sentences.split()
vocab = set(words)
vocab_size = len(vocab)
embed_dim = 10
context_size = 2
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
'''
The CBOW model tries to understand the context of the words and takes 
this as input. 
It then tries to predict words that are contextually accurate.
The model tries to predict the target word by trying to understand the
context of the surrounding words
'''
data = []
for i in range(2, len(words) - 2):
 context = [words[i - 2], words[i - 1], words[i + 1], words[i + 2]]
 target = words[i]
 data.append((context, target)) # With these word pairs, the model  
for i in range(0, 5):
 print(data[i])
(['we', 'are', 'to', 'study'], 'about')
(['are', 'about', 'study', 'the'], 'to')
(['about', 'to', 'the', 'idea'], 'study')
(['to', 'study', 'idea', 'of'], 'the')
(['study', 'the', 'of', 'a'], 'idea')
embeddings = np.random.random_sample((vocab_size, embed_dim)) #get 
def linear(m, theta):
 return m.dot(theta)
# softmax is used as the activation function for multi-class 

def log_softmax(x):
 e_x = np.exp(x - np.max(x)) # calculate exponential of all the 
 return np.log(e_x / e_x.sum())
# NLLLoss is a loss function commonly used in multi-classes 
def NLLLoss(logs, targets):
 out = logs[range(len(targets)), targets]
 return -out.sum()/len(out)
def log_softmax_crossentropy_with_logits(logits, target):
 out = np.zeros_like(logits) # returns an array of given shape and 
 out[np.arange(len(logits)), target] = 1
 softmax = np.exp(logits) / np.exp(logits).sum(axis = -1,keepdims =
True)
 return (-out + softmax) / logits.shape[0]
def forward(context_idxs, theta):
 m = embeddings[context_idxs].reshape(1, -1) # -1 means we dont 
 n = linear(m, theta)
 o = log_softmax(n)
 return m, n, o
def backward(preds, theta, target_idxs):
 m, n, o = preds
 dlog = log_softmax_crossentropy_with_logits(n, target_idxs)
 return m.T.dot(dlog)
def optimize(theta, grad, lr = 0.03):
 theta -= grad * lr
 return theta
theta = np.random.uniform(-1, 1, (2 * context_size * embed_dim, 
vocab_size))
# uniform(low, high, output_size)
# Samples are uniformly distributed over the half-open interval [low, 
# In other words, any value within the given interval is equally 
epoch_losses = {}
for epoch in range(80):
 losses = []
 for context, target in data:
  context_idxs = np.array([word_to_ix[w] for w in context])
 preds = forward(context_idxs, theta)
 target_idxs = np.array([word_to_ix[target]])
 loss = NLLLoss(preds[-1], target_idxs)
 losses.append(loss)
 grad = backward(preds, theta, target_idxs)
 theta = optimize(theta, grad, lr=0.03)
 epoch_losses[epoch] = losses
ix = np.arange(0,80)
plt.plot(ix,[epoch_losses[i][0] for i in ix])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Losses', fontsize=12)
def predict(words):
 context_idxs = np.array([word_to_ix[w] for w in words])
 preds = forward(context_idxs, theta)
 word = ix_to_word[np.argmax(preds[-1])]
 return word
print("['we', 'are', 'to', 'study'] : ", predict(['we', 'are', 'to', 
'study']))
print("['are', 'about', 'study', 'the'] : ",predict(['are', 'about', 
'study', 'the']))
print("['study', 'the', 'of', 'a'] : ",predict(['study', 'the', 'of', 
'a']))
