## print out all the node names ; edit the GRAPH_PB_PATH as per your exported grap
## This helps debug if we get any error after exporting
## My goal was to convert the frozen_inference_graph into Intel OpenVino and Nvidia TensorRT format
## But, I was getting errors. So, I used this script to debug the same.
import tensorflow as tf

GRAPH_PB_PATH = '/content/drive/MyDrive/SSD_EXPERIMENT5/output_inf_graph_SSD_FPN_COLAB_v2/frozen_inference_graph.pb'
with tf.Session() as sess:
   print("load graph")
   with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.compat.v1.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   with open("/content/drive/MyDrive/SSD_EXPERIMENT5/output_inf_graph_SSD_FPN_COLAB_v2/transfer.txt",'a') as fileWriter:
    for t in graph_nodes:
        names.append(t.name)
        print(t.name)
        fileWriter.write(t.name+"\n")
