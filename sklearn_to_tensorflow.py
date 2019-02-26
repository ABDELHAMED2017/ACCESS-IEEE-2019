import pickle
from os import path
import numpy as np
from sklearn.neural_network import MLPClassifier


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(x, inference=False):
    if inference:
        return np.argmax(x)
    else:
        return np.argmax(np.random.multinomial(1, x))

def sample_action(X, weights, inference=False):
    weights_1_n = D * h_1_size
    bias_1_n = h_1_size
    weights_2_n = h_1_size * h_2_size
    bias_2_n = h_2_size
    weights_3_n = h_2_size * K
    bias_3_n = K

    assert weights.shape[0] == (weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n + bias_3_n)
    assert X.shape[0] == D

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

    z_1 = np.tanh(weights_1.T.dot(X) + bias_1)
    z_2 = np.tanh(weights_2.T.dot(z_1) + bias_2)
    z_3 = weights_3.T.dot(z_2) + bias_3
    Y = softmax(z_3)
    action = predict(Y, inference=inference)

    return action

def get_splits(min_val, max_val, splits):
    delta = (max_val - min_val) / splits
    return min_val + delta/2 + delta * np.arange(splits)

def distill_policy(network_model, splits = 5, pickled_file = "refactored_model.p"):
    global max_bits, battery_cap

    # from itertools import product

    min_length = 30
    max_length = 200

    pkt_length = get_splits(min_length, max_length, splits)
    pkt_priority = get_splits(0, 10, splits)
    time = get_splits(0, 24 * 60 * 60, splits)
    bits_left = get_splits(0, max_bits, splits)
    battery_left = get_splits(0, battery_cap, splits)

    # X = np.array(list(product(
    #     pkt_length,
    #     pkt_priority,
    #     time,
    #     bits_left,
    #     battery_left
    # )))

    X = list()
    Y = list()

    num_vars = pkt_length.shape[0] * pkt_priority.shape[0] * time.shape[0] * bits_left.shape[0] * battery_left.shape[0]
    num_processed = 0
    check_every = 10000

    for pl in pkt_length:
        for pp in pkt_priority:
            for t in time:
                for bitsl in bits_left:
                    for batl in battery_left:
                        X.append(np.array([pl / max_length,
                                           t / sim_time,
                                           pp,
                                           bitsl / max_bits,
                                           batl / battery_cap]))

                        x = np.array([
                            pl / max_length,
                            t / sim_time,
                            pp,
                            bitsl / max_bits,
                            0,
                            0,
                            batl / battery_cap
                        ])
                        action = sample_action(x, network_model)



                        Y.append(action)
                        num_processed += 1
                        if num_processed % check_every == 0:
                            print("Processed {} out of {}".format(num_processed, num_vars))

    print("Fitting SVM on {} samples...".format(len(Y)))

    clf = MLPClassifier(solver='adam', alpha=1e-4, learning_rate='adaptive', shuffle=True, activation='tanh',
                        hidden_layer_sizes=(h_1_size_p, h_2_size_p), random_state=1, max_iter=int(10e6))
    clf.fit(X, np.array(Y, dtype=np.uint8))
    pickle.dump(clf, open(pickled_file, "wb"))

D = 7  # lambdas_.shape[0] * 2 + 1  #
K = 3
h_1_size = 45  # 500
h_2_size = 15  # 200
NPARAMS = (D + 1) * h_1_size + (h_1_size + 1) * h_2_size + (h_2_size + 1) * K

max_bits = 0.1e6  # 1Mb a day, justificado con tarifas ordinarias de provedores
sf = 7
cr = 5
speed_5g = 100e3  # 100kbps, asumimos 5G dividido en bandas
sim_time = 3600 * 24  # 1 day
consumption_5g = 2.15  # justificado con articulo, in Watts
consumption_lora = 0.1353  # justificado con articulo, in Watts (41 mA at 3.3V)
battery_cap = 2 * 15390 / 365 / 4  # justificado con web, two AA batteries designed to last for 3 years
h_1_size_p = 500
h_2_size_p = 200

model_path = 'models_100k/ES1.1s_model-20190206-120959.p'
# ES1.13s_model-20190206-132345.p
# ES1.2s_model-20190206-142043.p
# ES1.3s_model-20190206-150138.p
# ES1.4s_model-20190206-153330.p
# ES1.5s_model-20190206-155921.p
# ES1.62s_model-20190206-162127.p


model_large_path = ".".join(model_path.split('.')[:-1]) + "-large.p"

model = pickle.load(open(model_path, 'rb'))
assert model.shape[0] == NPARAMS

if not path.exists("refactored_model.p"):
    distill_policy(model, splits=9)

if path.exists("refactored_model.p") and not path.exists(model_large_path):
    weights = []
    clf = pickle.load(open('refactored_model.p', 'rb'))

    weights.extend(clf.coefs_[0].flatten())
    weights.extend(clf.intercepts_[0])

    weights.extend(clf.coefs_[1].flatten())
    weights.extend(clf.intercepts_[1])

    weights.extend(clf.coefs_[2].flatten())
    weights.extend(clf.intercepts_[2])

    weights = np.array(weights)
    print("Dumped weights to ", model_large_path)
    pickle.dump(weights, open(model_large_path, 'wb'))

model_large = np.array(pickle.load(open(model_large_path, 'rb')))
D = 5
K = 3
h_1_size = 500
h_2_size = 200
NPARAMS = (D + 1) * h_1_size + (h_1_size + 1) * h_2_size + (h_2_size + 1) * K
assert model_large.shape[0] == NPARAMS

print(sample_action(np.array([0.11, 0.28935185185185186, 1.5, 0.6, 0.5217673814165041]), model_large, inference=True))
print(sample_action(np.array([0.11, 0.28935185185185186, 0.3, 0.6, 0.5217673814165041]), model_large, inference=True))