import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import tensorflow as tf

import pickle
import os
import random

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device=", device)


def sample_generator(all_codnums, context_window_size, vocab_size, num_neg_samples):
    """ Form training pairs according to the skip-gram model. """
    for cod_nums in all_codnums:
        for i, center in enumerate(cod_nums):

            targets = cod_nums[max(0, i - context_window_size): i] + cod_nums[i + 1: i + context_window_size + 1]

            for target in targets:
                negatives = torch.zeros(num_neg_samples, dtype=torch.long).to(device)
                for j in range(num_neg_samples):
                    negative = center
                    while negative in targets:
                        negative = random.randint(0, vocab_size - 1)
                    negatives[j] = negative
                yield center, target, negatives


def batch_generator(sample_gen, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = torch.zeros(batch_size, dtype=torch.long).to(device)
        target_batch = torch.zeros(batch_size, dtype=torch.long).to(device)
        negative_batch = torch.zeros(batch_size, 5, dtype=torch.long).to(device)

        for i in range(batch_size):
            center_batch[i], target_batch[i], negative_batch[i] = next(sample_gen)
        yield center_batch, target_batch, negative_batch


def flatten(x):
    return [item for sublist in x for item in sublist]


def codnes_to_nums(cod, dictionary):
    return [dictionary[key] for key in cod]


def make_vocabulary_and_dictionary(all_codones):
    flat_codones = flatten(all_codones)
    vocab = set(flat_codones)
    dictionary = {cod: i for i, cod in enumerate(vocab)}
    return vocab, dictionary


def process_data(all_codnums, batch_size, skip_window, vocab_size, num_neg_samples):
    # but 3grams replaced by nums
    sample_gen = sample_generator(all_codnums, skip_window, vocab_size, num_neg_samples)
    batch_gen = batch_generator(sample_gen, batch_size=batch_size)
    return batch_gen  # returns batch of center_3gram_num --> target_3gram_num (target is context)


# all_codones = read_or_create(read_path='data/all_codones.pickle')
# vocab, codone_to_ix = make_vocabulary_and_dictionary(all_codones)


def save_all_codnums(batches, path):
    print('saving', path)
    with open(path, 'wb') as fp:
        pickle.dump(batches, fp)
    print("end of saving")


def read_all_codnums(read_path):
    print('reading', read_path)
    with open(read_path, 'rb') as fp:
        tmp = pickle.load(fp)
        print("end of reading")
        return tmp


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(NGramLanguageModeler, self).__init__()
        self.embed_center = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.embed_context = nn.Embedding(vocab_size, embedding_dim).to(device)

        self.embed_center.weight.data.uniform_(-0.2, 0.2)
        self.embed_context.weight.data.uniform_(-0.2, 0.2)

        # self.linear1 = nn.Linear(context_size * embedding_dim, 256).to(device)
        # self.linear2 = nn.Linear(256, vocab_size).to(device)

    def forward(self, center, target, negative):
        center_vec = self.embed_center(center)
        target_vec = self.embed_context(target)
        neg_vec = self.embed_context(negative)

        center_vec = center_vec.unsqueeze(2)
        target_vec = target_vec.unsqueeze(1)

        # print(center_vec.shape, target_vec.shape, neg_vec.shape)

        tar_score = -torch.bmm(target_vec, center_vec)
        tar_score = F.logsigmoid(tar_score)

        neg_score = torch.bmm(neg_vec, center_vec)
        neg_score = F.logsigmoid(neg_score)

        loss = -(tar_score + neg_score)
        # print(tar_score.shape, neg_score.shape, loss.shape)
        loss = loss.sum()

        return loss


def _main():
    BATCH_SIZE = 128
    SKIP_WINDOW = 12  # the context window
    learning_rate = 0.01
    EMBEDDING_DIM = 100
    NUM_NEG_SAMPLES = 5

    all_codnums = read_all_codnums("data/all_codnums.pickle")

    vocab_size = max(max(all_codnums))
    batch_gen = process_data(all_codnums, BATCH_SIZE, SKIP_WINDOW, vocab_size, NUM_NEG_SAMPLES)

    losses = []
    model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(10):
        total_loss = 0
        for i_batch, (center_batch, target_batch, negative_batch) in enumerate(batch_gen):
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = model(center_batch, target_batch, negative_batch)

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
            losses.append(loss.item())

            if i_batch % 100 == 0:
                print("batch", i_batch, "loss=", sum(losses[-100:]) / 100)
                losses = []
            # if i_batch % 1000 == 0:
            #     torch.save(model.state_dict(), "weights.pth")
            # print("\tbatch_loss=", loss.item())
        # print(losses)  # The loss decreased every iteration over the training data!

if __name__=="__main__":
    _main()