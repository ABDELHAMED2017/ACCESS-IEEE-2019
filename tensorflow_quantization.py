import pathlib
import pickle
from os import path
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

save_path = "./saved_model"

def load_weights(weights, D, K, h_1_size, h_2_size, NPARAMS):
    weights_1_n = D * h_1_size
    bias_1_n = h_1_size
    weights_2_n = h_1_size * h_2_size
    bias_2_n = h_2_size
    weights_3_n = h_2_size * K
    bias_3_n = K

    assert weights.shape[0] == (weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n + bias_3_n)

    weights_1 = weights[:weights_1_n]
    weights_1.shape = (D, h_1_size)
    bias_1 = weights[weights_1_n:weights_1_n + bias_1_n]
    weights_2 = weights[weights_1_n + bias_1_n:weights_1_n + bias_1_n + weights_2_n]
    weights_2.shape = (h_1_size, h_2_size)
    bias_2 = weights[weights_1_n + bias_1_n + weights_2_n:weights_1_n + bias_1_n + weights_2_n + bias_2_n]
    weights_3 = weights[
                weights_1_n + bias_1_n + weights_2_n + bias_2_n:weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n]
    weights_3.shape = (h_2_size, K)
    bias_3 = weights[weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n:]

    return weights_1, bias_1, weights_2, bias_2, weights_3, bias_3


D = 7  # lambdas_.shape[0] * 2 + 1  #
K = 3
h_1_size = 500# 45  # 1000
h_2_size = 200 #15  # 200
NPARAMS = (D + 1) * h_1_size + (h_1_size + 1) * h_2_size + (h_2_size + 1) * K

weights_1 = np.random.rand(D, h_1_size)
bias_1 = np.random.rand(h_1_size)
weights_2 = np.random.rand(h_1_size, h_2_size)
bias_2 = np.random.rand(h_2_size)
weights_3 = np.random.rand(h_2_size, K)
bias_3 = np.random.rand(K)

# model = pickle.load(open('models_100k/ES0.7s_model-20190206-083337.p', 'rb'))
# assert model.shape[0] == NPARAMS
# weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = load_weights(model, D, K, h_1_size, h_2_size, NPARAMS)

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, D])
hidden = layers.fully_connected(X, num_outputs=h_1_size, activation_fn=tf.nn.tanh, scope="fc/fc_1", weights_initializer=initializer)
hidden = layers.fully_connected(hidden, num_outputs=h_2_size, activation_fn=tf.nn.tanh, scope="fc/fc_2", weights_initializer=initializer)
logits = layers.fully_connected(hidden, num_outputs=K, activation_fn=None, scope="fc/fc_3", weights_initializer=initializer)
output = tf.nn.softmax(logits)
top = tf.argmax(output, 1)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()

    variables = tf.trainable_variables()
    weights_fc1 = [v for v in variables if "fc_1/weights" in v.name][0]
    biases_fc1 = [v for v in variables if "fc_1/biases" in v.name][0]
    weights_fc2 = [v for v in variables if "fc_2/weights" in v.name][0]
    biases_fc2 = [v for v in variables if "fc_2/biases" in v.name][0]
    weights_fc3 = [v for v in variables if "fc_3/weights" in v.name][0]
    biases_fc3 = [v for v in variables if "fc_3/biases" in v.name][0]

    sess.run([tf.assign(weights_fc1, weights_1),
              tf.assign(biases_fc1, bias_1),
              tf.assign(weights_fc2, weights_2),
              tf.assign(biases_fc2, bias_2),
              tf.assign(weights_fc3, weights_3),
              tf.assign(biases_fc3, bias_3)])

    x = np.array([
        [0.11, 0.28935185185185186, 1.5, 0.6, 0.0, 0.0, 0.5217673814165041],
        [0.11, 0.28935185185185186, 0.3, 0.6, 0.0, 0.0, 0.5217673814165041],
    ])
    best_action = sess.run([top], feed_dict={X: x})

    # tf.saved_model.simple_save(sess, save_path, inputs={"X": X}, outputs={"top": top})
    # print("Model saved in path: %s" % save_path)
    print(best_action)

    converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [X], [top])

    # converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
    # input_arrays = converter.get_input_arrays()
    # converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}  # mean, std_dev
    # converter.default_ranges_stats = (-16, 16)

    converter.post_training_quantize = True
    tflite_quant_model = converter.convert()
    open(path.join(save_path, "model_large_quant.tflite"), "wb").write(tflite_quant_model)

exit()

tflite_models_dir = pathlib.Path(save_path)
tflite_model_quant_file = tflite_models_dir/"model_quant.tflite"

# # CONVERT MODEL TO QUANTIZED VERSION!
# tf.enable_eager_execution()
# converter = tf.contrib.lite.TFLiteConverter.from_saved_model(save_path)
# tf.logging.set_verbosity(tf.logging.INFO)
# converter.post_training_quantize = True
# tflite_quant_model = converter.convert()
#
# tflite_model_quant_file.write_bytes(tflite_quant_model)
#

interpreter = tf.contrib.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

x = np.array([
    # [0.11, 0.29, 1.5, 0.6, 0.0, 0.0, 0.52],
    [0.11, 0.28935185185185186, 0.3, 0.6, 0.0, 0.0, 0.5217673814165041],
], dtype=np.float32)

interpreter.set_tensor(input_index, x)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

print(predictions)