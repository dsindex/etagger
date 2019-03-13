import sys, os, argparse
import tensorflow as tf
# for LSTMBlockFusedCell(), https://github.com/tensorflow/tensorflow/issues/23369
tf.contrib.rnn
# for QRNN
try: import qrnn
except: sys.stderr.write('import qrnn, failed\n')

'''
source is from https://gist.github.com/morgangiraud/249505f540a5e53a48b0c1a869d370bf#file-medium-tffreeze-1-py
'''

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, output_node_names, frozen_model_name, optimize_graph_def=0):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
        frozen_model_name: a string, the name of the frozen model
        optimize_graph_def: int, 1 for optimizing graph_def via tensorRT
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph_path = absolute_model_dir + "/" + frozen_model_name

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(',') # The output node names are used to select the usefull nodes
        )

        # Optimize graph_def via tensorRT
        if optimize_graph_def:
            from tensorflow.contrib import tensorrt as trt
            # get optimized graph_def
            trt_graph_def = trt.create_inference_graph(
              input_graph_def=output_graph_def,
              outputs=output_node_names.split(','),
              max_batch_size=128,
              max_workspace_size_bytes=1 << 30,
              precision_mode='FP16',  # TRT Engine precision "FP32","FP16" or "INT8"
              minimum_segment_size=3  # minimum number of nodes in an engine
            )
            output_graph_def = trt_graph_def 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Model folder to export", required=True)
    parser.add_argument("--frozen_model_name", type=str, help="The name of the frozen model", required=True)
    parser.add_argument("--output_node_names", type=str, help="The name of the output nodes, comma separated.", required=True)
    parser.add_argument("--optimize_graph_def", type=int, help="1 for optimizing graph_def via tensorRT, default 0", default=0, required=False)
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names, args.frozen_model_name, args.optimize_graph_def)

    
