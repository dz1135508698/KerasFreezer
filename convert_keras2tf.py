import tensorflow as tf
from keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, default=None, help='Keras model to be converted to pb.')
ap.add_argument('-o', '--out_name', required=False, default='keras2tf.pb', help='Name of the tensorflow pb file.')
ap.add_argument('-d', '--dir', required=True, default=None, help='Path to save pb file')
args = vars(ap.parse_args())

keras_model = './' + args['model']
model = load_model(keras_model)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



# save keras model as tf pb files ===============
from keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, args['dir'], args['out_name'], as_text=False)