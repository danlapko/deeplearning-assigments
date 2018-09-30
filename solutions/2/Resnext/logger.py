from io import BytesIO

import scipy.misc
import tensorflow as tf


class Logger(object):

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
