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
from io import BytesIO

import scipy.misc
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device=", device)


class Logger:
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        scalar_summaries = [tf.Summary.Value(tag=tag, simple_value=value)]
        summary = tf.Summary(value=scalar_summaries)
        self.writer.add_summary(summary, step)

    def image_summary(self, tags, images, step):
        img_summaries = []
        for tag, img in zip(tags, images):
            s = BytesIO()  # Write the image to a string
            scipy.misc.toimage(img).save(s, format="png")
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            img_summaries.append(tf.Summary.Value(tag=f"{tag}", image=img_sum))

        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)


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


def process_data(all_codnums, batch_size, skip_window, vocab_size, num_neg_samples):
    # 3grams replaced by nums
    sample_gen = sample_generator(all_codnums, skip_window, vocab_size, num_neg_samples)
    batch_gen = batch_generator(sample_gen, batch_size=batch_size)
    return batch_gen  # returns batches of center_3gram_num, target_3gram_num, 5 x negative_3gram_nums (num in terms ix in vocabulary)


def read_all_codnums(read_path):
    print('reading', read_path)
    with open(read_path, 'rb') as fp:
        tmp = pickle.load(fp)
        print("end of reading")
        return tmp


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim, batch_size):
        super(SkipGram, self).__init__()
        self.batch_size = batch_size

        self.embed_center = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.embed_context = nn.Embedding(vocab_size, embedding_dim).to(device)

        self.embed_center.weight.data.uniform_(-0.2, 0.2)
        self.embed_context.weight.data.uniform_(-0.2, 0.2)

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


def train(logger):
    BATCH_SIZE = 128
    SKIP_WINDOW = 12  # the context window
    learning_rate = 0.01
    EMBEDDING_DIM = 100
    NUM_NEG_SAMPLES = 5
    BATCHES_TO_PROCESS = 100_000
    PRINT_LOSS_EACH = 100
    SAVE_WEIGHTS_EACH = 1000
    WEIGHTS_PATH = "weights.pth"
    ALL_CODNUMS_PATH = "data/all_codnums.pickle"

    all_codnums = read_all_codnums(ALL_CODNUMS_PATH)

    vocab_size = max(max(all_codnums))
    batch_gen = process_data(all_codnums, BATCH_SIZE, SKIP_WINDOW, vocab_size, NUM_NEG_SAMPLES)

    losses = []
    model = SkipGram(vocab_size, EMBEDDING_DIM, BATCH_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("starting to train...")
    for i_batch in range(BATCHES_TO_PROCESS):

        center_batch, target_batch, negative_batch = next(batch_gen)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        loss = model(center_batch, target_batch, negative_batch)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        logger.scalar_summary('loss', losses[-1], i_batch)

        if i_batch % PRINT_LOSS_EACH == 0:
            print("batch", i_batch, f"loss= {sum(losses[-PRINT_LOSS_EACH:]) / PRINT_LOSS_EACH:.5}")

        if i_batch % SAVE_WEIGHTS_EACH == 0:
            torch.save(model.state_dict(), WEIGHTS_PATH)


def _main():
    logger = Logger('./logs')
    train(logger)


if __name__ == "__main__":
    _main()
