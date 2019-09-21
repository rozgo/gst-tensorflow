import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

with tf.Session() as sess:
    graph_def = tf.GraphDef()
    model_filename = "../models/deeplabv3_mnv2_dm05_pascal_trainval/frozen_inference_graph.pb"
    pb_file = open(model_filename,"rb")
    graph_def = tf.GraphDef.FromString(pb_file.read())
    pb_file.close()
    g_in = tf.import_graph_def(graph_def)

print("###########################################")
print("Type of ops")
print("...........................................")
type_of_ops = [op.type for op in sess.graph.get_operations()]
type_of_ops = np.unique(type_of_ops)
for type_of_op in type_of_ops:
    print(type_of_op)
print("...........................................")

print("###########################################")
print("Placeholders")
print("...........................................")
placeholders = [op for op in sess.graph.get_operations() if op.type == "Placeholder"]
for placeholder in placeholders:
    print(placeholder)
print("...........................................")

# print("###########################################")
# print("Identities")
# print("...........................................")
# indentities = [op for op in sess.graph.get_operations() if op.type == "Identity"]
# for indentity in indentities:
#     print(indentity)
# print("...........................................")

# LOGDIR="temp"
# train_writer = tf.summary.FileWriter(LOGDIR)
# train_writer.add_graph(sess.graph)
# train_writer.flush()
# train_writer.close()

