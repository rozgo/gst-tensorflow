import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    graph_def = tf.GraphDef()
    model_filename = "../models/deeplabv3_mnv2_pascal_train_aug_2018_01_29/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb"
    pb_file = open(model_filename,"rb")
    graph_def = tf.GraphDef.FromString(pb_file.read())
    pb_file.close()
    g_in = tf.import_graph_def(graph_def)
LOGDIR="temp"
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
