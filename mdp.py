import glob
import os
import pickle
import platform
import random
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from datetime import timedelta
from enum import Enum
from math import pow, ceil

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC

from estool.es import CMAES

global D, K, h_1_size, h_2_size, dataframe, max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora, battery_cap, threshold, THRESHOLD_PRIORITY

if platform.node() == "alioth":
    print('Alioth config')
    NPOPULATION = 60  # use population size of 101.
    WORKERS = 39
    REPS = 40
    MAX_ITERATION = 5000
    OFF_PERIOD = False
else:
    print('Laptop config')
    NPOPULATION = 60  # use population size of 101.
    WORKERS = 1
    REPS = 40
    MAX_ITERATION = 5000
    OFF_PERIOD = False


class Actions(Enum):
    LORA = 0
    FIVEG = 1
    DROP = 2


# LORA = 0 -> 1
# FIVEG = 1 -> 2
# DROP = 2 -> 0

nn_to_tree = {0: 1, 1: 2, 2: 0}
tree_to_nn = {0: 2, 1: 0, 2: 1}


def get_reward(length, priority, delay):
    # length in bytes
    # delay in seconds
    return length * 8 * priority / delay


def get_lora_delay(payload_length, sf, cr, overhead=13):
    # payload_length in bytes
    BW = 125e3
    preamble_symbols = 8
    header_length = overhead
    explicit_header = 1

    if sf == 0 and cr == 0:
        return 0

    assert 7 <= sf <= 12
    assert 5 <= cr <= 7
    de = 1 if sf >= 11 else 0
    # http://forum.thethingsnetwork.org/t/spreadsheet-for-lora-airtime-calculation/1190/15
    t_sym = pow(2, sf) / BW * 1000  # symbol time in ms
    t_preamble = (preamble_symbols + 4.25) * t_sym  # over the air time of the preamble
    payload_symbol_number = 8 + max([(ceil(
        (8 * (payload_length + header_length) - 4 * sf + 28 + 16 - 20 * (1 - explicit_header)) / (
            4 * (sf - 2 * de))) * cr), 0])  # number of symbols of the payload

    t_payload = payload_symbol_number * t_sym  # payload time in ms
    t_packet = t_preamble + t_payload

    return t_packet / 1000  # expressed in seconds


def get_5g_delay(payload_length, speed, overhead):
    # payload_length in bytes
    # speed in bits per second (bps)

    return (payload_length + overhead) * 8 / speed


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict(x):
    return np.argmax(np.random.multinomial(1, x))


def sample_action(X, weights):
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
    action = predict(Y)

    return action


def priority_based_rollout(model, df, max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora,
                           battery_cap):
    global THRESHOLD_PRIORITY

    t = 0
    left_bits = max_bits
    total_reward = 0
    min_length = 30
    max_length = 200  # in bytes
    overhead_5G = 50  # in bytes
    battery_left = battery_cap
    stop_lora = 0
    stop_5g = 0
    total_usages = [0 for _ in range(len(Actions.__members__))]
    actions = list()

    while True:
        try:
            next_row = df.__next__()[1]
        except StopIteration:
            break

        next_event = next_row["timestamp"]
        next_value = next_row["value_hrf"]

        t = next_event

        pkt_length = int(np.random.uniform(min_length, max_length))  # in bytes
        pkt_priority = next_value

        if battery_left <= 0:
            action = Actions.DROP.value
            reward = 0
            total_usages[Actions.DROP.value] += 1
        else:
            delay_lora = max(stop_lora - t, 0)
            delay_5g = max(stop_5g - t, 0)

            # action = int(pkt_priority > 0.3)
            # if pkt_priority < 0.1:
            #     action = Actions.DROP.value
            action = int(pkt_priority > THRESHOLD_PRIORITY)

            if action == Actions.FIVEG.value:
                if left_bits < ((pkt_length + overhead_5G) * 8):
                    action = Actions.LORA.value
                else:
                    left_bits -= ((pkt_length + overhead_5G) * 8)

            if action == Actions.LORA.value:
                tx_time = get_lora_delay(pkt_length, sf, cr)
                stop_lora = delay_lora + t + tx_time
                reward = get_reward(pkt_length, pkt_priority, stop_lora - t)
                if OFF_PERIOD:
                    stop_lora += (tx_time / 0.1 - tx_time)
                battery_left -= (consumption_lora * tx_time)
                total_usages[Actions.LORA.value] += 1
            elif action == Actions.FIVEG.value:
                tx_time = get_5g_delay(pkt_length, speed_5g, overhead_5G)
                stop_5g = delay_5g + t + tx_time
                reward = get_reward(pkt_length, pkt_priority, stop_5g - t)
                battery_left -= (consumption_5g * tx_time)
                total_usages[Actions.FIVEG.value] += 1
            elif action == Actions.DROP.value:
                reward = 0
                total_usages[Actions.DROP.value] += 1
            else:
                raise Exception('Non defined action')

        actions.append((pkt_priority, pkt_length, action))
        total_reward += reward

    # if battery_left <= 0:
    #     print('Battery ended at {}%'.format(t / sim_time))
    # print('Sim ended')

    return total_reward, total_usages, actions


def five_g_first_rollout(model, df, max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora,
                         battery_cap):
    t = 0
    left_bits = max_bits
    total_reward = 0
    min_length = 30
    max_length = 200  # in bytes
    overhead_5G = 50  # in bytes
    battery_left = battery_cap
    stop_lora = 0
    stop_5g = 0
    total_usages = [0 for _ in range(len(Actions.__members__))]
    actions = list()

    while True:
        try:
            next_row = df.__next__()[1]
        except StopIteration:
            break

        next_event = next_row["timestamp"]
        next_value = next_row["value_hrf"]

        t = next_event

        pkt_length = int(np.random.uniform(min_length, max_length))  # in bytes
        pkt_priority = next_value

        if battery_left <= 0:
            action = Actions.DROP.value
            reward = 0
            total_usages[Actions.DROP.value] += 1
        else:
            delay_lora = max(stop_lora - t, 0)
            delay_5g = max(stop_5g - t, 0)

            if left_bits >= ((pkt_length + overhead_5G) * 8):
                action = Actions.FIVEG.value
                left_bits -= ((pkt_length + overhead_5G) * 8)
            else:
                action = Actions.LORA.value

            if action == Actions.LORA.value:
                tx_time = get_lora_delay(pkt_length, sf, cr)
                stop_lora = delay_lora + t + tx_time
                reward = get_reward(pkt_length, pkt_priority, stop_lora - t)
                if OFF_PERIOD:
                    stop_lora += (tx_time / 0.1 - tx_time)
                battery_left -= (consumption_lora * tx_time)
                total_usages[Actions.LORA.value] += 1
            elif action == Actions.FIVEG.value:
                tx_time = get_5g_delay(pkt_length, speed_5g, overhead_5G)
                stop_5g = delay_5g + t + tx_time
                reward = get_reward(pkt_length, pkt_priority, stop_5g - t)
                battery_left -= (consumption_5g * tx_time)
                total_usages[Actions.FIVEG.value] += 1
            elif action == Actions.DROP.value:
                reward = 0
                total_usages[Actions.DROP.value] += 1
            else:
                raise Exception('Non defined action')

        actions.append((pkt_priority, pkt_length, action))
        total_reward += reward

    # if battery_left <= 0:
    #     print('Battery ended at {}%'.format(t / sim_time))
    # print('Sim ended')

    return total_reward, total_usages, actions


def random_rollout(model, df, max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora, battery_cap):
    t = 0
    left_bits = max_bits
    total_reward = 0
    min_length = 30
    max_length = 200  # in bytes
    overhead_5G = 50  # in bytes
    battery_left = battery_cap
    stop_lora = 0
    stop_5g = 0
    total_usages = [0 for _ in range(len(Actions.__members__))]
    actions = list()

    while True:
        try:
            next_row = df.__next__()[1]
        except StopIteration:
            break

        next_event = next_row["timestamp"]
        next_value = next_row["value_hrf"]

        t = next_event

        pkt_length = int(np.random.uniform(min_length, max_length))  # in bytes
        pkt_priority = next_value

        if battery_left <= 0:
            action = Actions.DROP.value
            reward = 0
            total_usages[Actions.DROP.value] += 1
        else:
            delay_lora = max(stop_lora - t, 0)
            delay_5g = max(stop_5g - t, 0)

            action = int(random.random() > 0.5)

            if action == Actions.FIVEG.value:
                if left_bits < ((pkt_length + overhead_5G) * 8):
                    action = Actions.DROP.value
                else:
                    left_bits -= ((pkt_length + overhead_5G) * 8)
            if action == Actions.LORA.value:
                tx_time = get_lora_delay(pkt_length, sf, cr)
                stop_lora = delay_lora + t + tx_time
                reward = get_reward(pkt_length, pkt_priority, stop_lora - t)
                if OFF_PERIOD:
                    stop_lora += (tx_time / 0.1 - tx_time)
                battery_left -= (consumption_lora * tx_time)
                total_usages[Actions.LORA.value] += 1
            elif action == Actions.FIVEG.value:
                tx_time = get_5g_delay(pkt_length, speed_5g, overhead_5G)
                stop_5g = delay_5g + t + tx_time
                reward = get_reward(pkt_length, pkt_priority, stop_5g - t)
                battery_left -= (consumption_5g * tx_time)
                total_usages[Actions.FIVEG.value] += 1
            elif action == Actions.DROP.value:
                reward = 0
                total_usages[Actions.DROP.value] += 1
            else:
                raise Exception('Non defined action')

        actions.append((pkt_priority, pkt_length, action))
        total_reward += reward

    # if battery_left <= 0:
    #     print('Battery ended at {}%'.format(t / sim_time))
    # print('Sim ended')

    return total_reward, total_usages, actions


def rollout(model, df, max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora, battery_cap):
    # sim_time = 1 day

    # t = 0
    left_bits = max_bits
    total_reward = 0
    min_length = 30
    max_length = 200  # in bytes
    overhead_5G = 50  # in bytes
    battery_left = battery_cap
    stop_lora = 0
    stop_5g = 0
    total_usages = [0 for _ in range(len(Actions.__members__))]
    actions = list()

    # trained_model = pickle.load(open("../refactored_model.p", "rb"))


    while True: # iterate until end of dataframe iterator
        try:
            next_row = df.__next__()[1]
        except StopIteration:
            break

        next_event = next_row["timestamp"]
        next_value = next_row["value_hrf"]

        # next_event = - np.log(1.0 - np.random.rand()) / lambda_
        # t += next_event

        t = next_event

        pkt_length = int(np.random.uniform(min_length, max_length))  # in bytes
        pkt_priority = next_value # np.random.random()

        if battery_left <= 0:
            action = Actions.DROP.value
            reward = 0
        else:
            delay_lora = max(stop_lora - t, 0)
            delay_5g = max(stop_5g - t, 0)

            observation = np.array([pkt_length / max_length,
                                    t / sim_time,
                                    pkt_priority,
                                    left_bits / max_bits,
                                    delay_lora,
                                    delay_5g,
                                    battery_left / battery_cap,
                                    ])

            action = sample_action(observation, model)

            # observation = np.array([pkt_length / max_length,
            #                         t / sim_time,
            #                         pkt_priority,
            #                         left_bits / max_bits,
            #                         battery_left / battery_cap,
            #                         ])
            #
            # action = trained_model.predict(observation.reshape(1, -1))[0]


            # asdf
            # if action == Actions.DROP.value:
            #     if pkt_priority > 1.05:
            #         action = Actions.LORA.value

            if action == Actions.FIVEG.value:
                if left_bits < ((pkt_length + overhead_5G) * 8):
                    action = Actions.DROP.value
                    # print('Bits exhausted for 5G')
                else:
                    left_bits -= ((pkt_length + overhead_5G) * 8)

            if action == Actions.LORA.value:
                tx_time = get_lora_delay(pkt_length, sf, cr)
                stop_lora = delay_lora + t + tx_time
                reward = get_reward(pkt_length, pkt_priority, stop_lora - t)
                if OFF_PERIOD:
                    stop_lora += (tx_time / 0.1 - tx_time)
                battery_left -= (consumption_lora * tx_time)
                total_usages[Actions.LORA.value] += 1
            elif action == Actions.FIVEG.value:
                tx_time = get_5g_delay(pkt_length, speed_5g, overhead_5G)
                stop_5g = delay_5g + t + tx_time
                reward = get_reward(pkt_length, pkt_priority, stop_5g - t)
                battery_left -= (consumption_5g * tx_time)
                total_usages[Actions.FIVEG.value] += 1
            elif action == Actions.DROP.value:
                # if left_bits >= ((pkt_length + overhead_5G) * 8):   # 5G could have been used
                #     tx_time_5g = get_5g_delay(pkt_length, speed_5g, overhead_5G)
                #     stop_5g_p = delay_5g + tx_time_5g
                #     reward_5g = get_reward(pkt_length, pkt_priority, stop_5g_p)
                #     reward = - (reward_5g * 0.1)
                # else:
                #     tx_time_lora = get_lora_delay(pkt_length, sf, cr)
                #     stop_lora_p = delay_lora + tx_time_lora
                #     reward_lora = get_reward(pkt_length, pkt_priority, stop_lora_p)
                #
                #     reward = - (reward_lora * 0.1)

                reward = 0
                total_usages[Actions.DROP.value] += 1
            else:
                raise Exception('Non defined action')

        actions.append((pkt_priority, pkt_length, action))
        total_reward += reward

    # if battery_left <= 0:
    #     print('Battery ended')
    # else:
    #     print("Battery not ended", battery_left / battery_cap * 100)

    # print('Sim ended')

    return total_reward, total_usages, actions


def rollout_rep(params):
    global max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora, battery_cap
    model, df = params

    # random.seed(seed)
    # np.random.seed(seed)

    rs = list()
    for rep in range(REPS):
        r, _, _ = rollout(model, df[rep].iterrows(), max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora,
                          battery_cap)
        rs.append(r)
    rs = np.array(rs)

    return rs


def rollout_evaluate(params):
    global max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora, battery_cap
    algo, model, df = params

    # random.seed(seed)
    # np.random.seed(seed)

    rs = list()
    total_usages = np.zeros(3)
    total_actions = list()
    for rep in range(REPS):
        # r, usages, actions = algo(model, df[rep].iterrows(), max_bits, sf, cr, speed_5g, sim_time, consumption_5g,
        r, usages, actions = algo(model, df.iterrows(), max_bits, sf, cr, speed_5g, sim_time, consumption_5g,
                                  consumption_lora,
                                  battery_cap)
        total_usages += usages
        rs.append(r)
        total_actions.extend(actions)
    rs = np.array(rs)

    return rs, total_usages, total_actions


def evaluate(model, algo, dataframe):
    performances = []
    total_usages = []
    total_actions = []
    for _ in range(MAX_ITERATION):
        v, usages, actions = rollout_evaluate((algo, model, dataframe[_]))
        total_usages.append(usages)
        performances.extend(v.tolist())
        total_actions.extend(actions)

    usgs = np.array(total_usages)
    s = usgs.sum(axis=1)
    usgs_norm = usgs / np.tile(s, (3, 1)).T

    mean = np.mean(usgs_norm, axis=0)
    std = np.std(usgs_norm, axis=0)
    print('Done evaluating...')

    return np.mean(performances), np.std(performances), mean, std, total_actions



def timestamp_to_epoch(ts):
    return timestamp_to_datetime(ts).timestamp()


def timestamp_to_datetime(ts):
    p = '%Y/%m/%d %H:%M:%S'
    return datetime.strptime(ts, p)

def epoch_to_datetime(ts):
    return datetime.fromtimestamp(ts)

def get_random_days(dataframe, days):
    return [random.choice(dataframe) for _ in range(days)]


def slice_df_in_days(df, offsets):
    start = offsets[0][0]
    if len(offsets) == 1:
        end = len(df)
    else:
        end = offsets[1][0]

    slices = []

    df_p = df.iloc[start:end, :]
    base_date = df_p.iloc[0, 1]
    # df_p["timestamp"] = df_p["timestamp"].apply(lambda x: (x - base_date).total_seconds())
    df_p.loc[:, "timestamp"] = df_p["timestamp"].apply(lambda x: (x - base_date).total_seconds())
    slices.append(df_p)

    for offset_i in range(1, len(offsets)):
        start = offsets[offset_i][0]
        if offset_i + 1 < len(offsets):
            end = offsets[offset_i + 1][0]
        else:
            end = len(df)

        df_p = df.iloc[start:end, :]
        base_date = df_p.iloc[0, 1]

        df_p.loc[:, "timestamp"] = df_p["timestamp"].apply(lambda x: (x - base_date).total_seconds())

        slices.append(df_p)

    return slices[:-1]  # the last slice is incomplete

def test_solver(solver):
    # warnings.warn('{} reps solo'.format(REPS))

    average_performance = []
    for j in range(MAX_ITERATION):
        solutions = solver.ask()

        # seeds = np.random.randint(0, 4294967295, solutions.shape[0])
        # seeds = np.full((solutions.shape[0]), j)
        dfs = get_random_days(dataframe, REPS)
        dfs = [dfs] * solutions.shape[0]

        if WORKERS > 1:
            with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                p = executor.map(rollout_rep, zip(solutions, dfs))
                fitness_list = np.array(list(p))
        else:
            fitness_list = list()
            for i in range(solutions.shape[0]):
                fitness_list.append(
                    rollout_rep([solutions[i], dfs[i]]))  # rollout(environment, solutions[i], D, K, h_1_size)

        fitness_list_means = []
        for fl in fitness_list:
            v = np.mean(fl)
            fitness_list_means.append(v)

        solver.tell(fitness_list_means)
        result = solver.result()  # first element is the best solution, second element is the best fitness

        # print results
        average_performance.append(np.mean(fitness_list_means))
        print("Max fitness at iteration {} is {}.\nAverage fittness at iteration {}".
              format((j + 1), result[1], average_performance[-1]))

        pickle.dump(average_performance, open('average_performance_ES{:03}s.p'.format(threshold), 'wb'))

        if (j + 1) % 10 == 0:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            pickle.dump(result[0], open('models/ES{:03}s_model-{}.p'.format(threshold, timestr), 'wb'))
            # print('it done')


def show_save_actions(actions, ia, policy, num=1000):
    # pickle.dump(total_actions, open('total_actions.p', 'wb'))
    chosen_actions = random.sample(actions, num)
    chosen_lora_actions = np.array([a for a in chosen_actions if a[2] == Actions.LORA.value])
    chosen_5G_actions = np.array([a for a in chosen_actions if a[2] == Actions.FIVEG.value])
    chosen_drop_actions = np.array([a for a in chosen_actions if a[2] == Actions.DROP.value])

    legend = []

    if chosen_5G_actions.shape[0] > 0:
        plt.plot(chosen_5G_actions[:, 1], chosen_5G_actions[:, 0], 'b*', markersize='4')
        legend.append('5G')

    if chosen_lora_actions.shape[0] > 0:
        plt.plot(chosen_lora_actions[:, 1], chosen_lora_actions[:, 0], 'ro', markersize='4')
        legend.append('LoRa')

    if chosen_drop_actions.shape[0] > 0:
        plt.plot(chosen_drop_actions[:, 1], chosen_drop_actions[:, 0], 'k+', markersize='4')
        legend.append('Drop')

    plt.legend(legend, loc=1)
    plt.xlabel('Packet length (bytes)')
    plt.ylabel('Packet priority')
    plt.title('Distribution of RAT vs packet length and priority.\n {} policy'.format(policy))
    plt.savefig('actions_{}_{}.png'.format(policy, ia), dpi=600)
    plt.clf()
    print('Done exporting {} policy with ia={}'.format(policy, ia))

    # plt.show()

def get_splits(min_val, max_val, splits):
    delta = (max_val - min_val) / splits
    return min_val + delta/2 + delta * np.arange(splits)


def get_offsets(df, period):
    assert isinstance(period, timedelta)

    offsets = list()
    from_ts = df.iloc[0, 1]
    ptr = 0
    offsets.append((ptr, from_ts))

    from_ts += period

    for ptr in range(len(df)):
        if df.iloc[ptr, 1] >= from_ts:
            offsets.append((ptr, from_ts))
            from_ts += period

    return offsets


def show_rat_usages(usages):
    assert len(usages) == 3


def get_threshold(model_name):
    m = re.search(r'ES([0-9\.]+)s', str(model_name))
    threshold = float(m.group(1))  # int(m.group(1))
    return threshold

def foo():
    print("bar")

if __name__ == '__main__':
    global D, K, h_1_size, h_2_size, dataframe, max_bits, sf, cr, speed_5g, sim_time, consumption_5g, consumption_lora, battery_cap, threshold, THRESHOLD_PRIORITY

    D = 7  # lambdas_.shape[0] * 2 + 1  #
    K = 3
    h_1_size = 45  # 100  # 20
    h_2_size = 15  # 50  # 10
    NPARAMS = (D + 1) * h_1_size + (h_1_size + 1) * h_2_size + (h_2_size + 1) * K

    random.seed(14)
    np.random.seed(14)

    print('Selecting last generated model')
    models = glob.glob('models_100k/*model*.p')
    if len(models) > 0:
        models.sort(key=os.path.getmtime)
        model_path = models[-1]
        print('Found model to be loaded:', model_path, '\nPress any key to continue.')
        # input()
        model = pickle.load(open(model_path,
                                 'rb'))
        assert model.shape[0] == NPARAMS
    else:
        model = None

    cmaes = CMAES(NPARAMS,
                  popsize=NPOPULATION,
                  sigma_init=1,
                  x0=model,
                  weight_decay=0
                  )

    # pepg = PEPG(num_params=NPARAMS, popsize=NPOPULATION)
    # openes = OpenES(num_params=NPARAMS, popsize=NPOPULATION, antithetic=True)

    # lambda_ = 1 / (45)  # digo que justifico con articulo (Smart city wireless ... cost analysis

    max_bits = 0.1e6  # 1Mb a day, justificado con tarifas ordinarias de provedores
    sf = 7
    cr = 5
    speed_5g = 100e3  # 100kbps, asumimos 5G dividido en bandas
    sim_time = 3600 * 24  # 1 day
    consumption_5g = 2.15  # justificado con articulo, in Watts
    consumption_lora = 0.1353  # justificado con articulo, in Watts (41 mA at 3.3V)
    battery_cap = 2 * 15390 / 365 / 4  # justificado con web, two AA batteries designed to last for 3 years

    # An Accurate Measurement-Based Power Consumption Model for LTE Uplink Transmissions
    # Modeling the Energy Performance of LoRaWAN
    # battery: https://hypertextbook.com/facts/2001/KhalidaNisimova.shtml
    # Smart City Wireless Connectivity Considerations and Cost Analysis: Lessons Learnt From Smart Water Case Studies
    #
    # http://www.easym2m.eu

    do_evaluation = True
    if do_evaluation:
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D

        foo()
        # root_folder = 'sin off-period/'
        root_folder = 'models_100k/'
        OFF_PERIOD = False

        os.chdir(root_folder)

        models = glob.glob('ES*_model-*.p')
        inter_arrivals = []
        xticks = []
        mean_rewards_opt = []
        mean_rewards_rnd = []
        mean_rewards_5gf = []
        mean_rewards_pri = []

        std_rewards_opt = []
        std_rewards_rnd = []
        std_rewards_5gf = []
        std_rewards_pri = []

        mean_usages_opt = []
        mean_usages_rnd = []
        mean_usages_5gf = []
        mean_usages_pri = []

        MAX_ITERATION = 100
        REPS = 1

        models_thresholds = {model: get_threshold(model) for model in models}

        for model, threshold in sorted(models_thresholds.items(), key=lambda x: x[1], reverse=False):
            random.seed(14)
            np.random.seed(14)

            foo()

            threshold = get_threshold(model)
            assert threshold > 0
            print("Loading dataframe for", model)
            model = pickle.load(open(model, 'rb'))

            # model = 0.125 * np.round(model / 0.125) # model // 0.125
            # model_quantized = (np.round(model / 0.125) + 128).astype(np.uint8)

            warnings.warn("Cambiar a june.csv")

            dataframe = pd.read_csv("../data_first100k.csv", header=None, names=["node_id", "timestamp", "value_hrf"])
            dataframe["timestamp"] = dataframe["timestamp"].apply(timestamp_to_datetime)

            dataframe = dataframe[dataframe["value_hrf"] > threshold]

            THRESHOLD_PRIORITY = np.mean(dataframe.iloc[:, 2])

            # First, apply threshold, then compute thresholds
            offsets = get_offsets(dataframe, timedelta(days=1, hours=0, minutes=0, seconds=0))
            print("Slicing data frame")
            dataframe = slice_df_in_days(dataframe, offsets)
            ia = int(round(1/np.mean([len(v)/(3600*24) for v in dataframe])))
            print('Evaluating model with IA={}'.format(ia))

            dataframe = get_random_days(dataframe, MAX_ITERATION)
            # foo()

            inter_arrivals.append(ia)
            xticks.append(r'$\frac{1}{' + str(ia) + '}$')

            mean_reward_opt, std_reward_opt, mean_usage_opt, stsd_usage_opt, actions = evaluate(model, rollout, dataframe)
            mean_rewards_opt.append(mean_reward_opt)
            std_rewards_opt.append(std_reward_opt)
            mean_usages_opt.append(mean_usage_opt)
            show_save_actions(actions, ia, 'Proposed')

            random.seed(14)
            np.random.seed(14)

            print(mean_reward_opt)

            mean_reward_pri, std_reward_pri, mean_usage_pri, stsd_usage_pri, actions = evaluate(model, priority_based_rollout, dataframe)
            mean_rewards_pri.append(mean_reward_pri)
            std_rewards_pri.append(std_reward_pri)
            mean_usages_pri.append(mean_usage_pri)
            show_save_actions(actions, ia, 'Priority-based')

            random.seed(14)
            np.random.seed(14)

            print(mean_reward_pri)

            mean_reward_5gf, std_reward_5gf, mean_usage_5gf, stsd_usage_5gf, actions = evaluate(model, five_g_first_rollout, dataframe)
            mean_rewards_5gf.append(mean_reward_5gf)
            std_rewards_5gf.append(std_reward_5gf)
            mean_usages_5gf.append(mean_usage_5gf)
            show_save_actions(actions, ia, '5G First')

            random.seed(14)
            np.random.seed(14)

            print(mean_reward_5gf)

            mean_reward_rnd, std_reward_rnd, mean_usage_rnd, stsd_usage_rnd, actions = evaluate(model, random_rollout, dataframe)
            mean_rewards_rnd.append(mean_reward_rnd)
            std_rewards_rnd.append(std_reward_rnd)
            mean_usages_rnd.append(mean_usage_rnd)
            show_save_actions(actions, ia, 'Random')

            random.seed(14)
            np.random.seed(14)

            print(mean_reward_rnd)
            exit()


        print(mean_rewards_opt)
        print(mean_rewards_5gf)
        print(mean_rewards_pri)
        print(mean_rewards_rnd)

        print(mean_usages_opt)
        print(mean_usages_5gf)
        print(mean_usages_pri)
        print(mean_usages_rnd)

        inter_arrivals = 1 / np.array(inter_arrivals)

        plt.figure()

        # plt.errorbar(inter_arrivals, mean_rewards_opt, yerr=std_rewards_opt[::-1], fmt='r', capsize=4)
        # plt.errorbar(inter_arrivals, mean_rewards_5gf, yerr=std_rewards_5gf[::-1], fmt='k', capsize=4)
        # plt.errorbar(inter_arrivals, mean_rewards_pri, yerr=std_rewards_pri[::-1], fmt='c', capsize=4)
        # plt.errorbar(inter_arrivals, mean_rewards_rnd, yerr=std_rewards_rnd[::-1], fmt='b', capsize=4)

        plt.plot(inter_arrivals, mean_rewards_opt, color='r', linewidth=4)
        plt.plot(inter_arrivals, mean_rewards_5gf, color='k', linewidth=4)
        plt.plot(inter_arrivals, mean_rewards_pri, color='c', linewidth=4)
        plt.plot(inter_arrivals, mean_rewards_rnd, color='b', linewidth=4)

        plt.xticks(inter_arrivals, xticks)

        legend_elements = [
            Line2D([0], [0], color='r', lw=2, label='Proposed policy'),
            Line2D([0], [0], color='k', lw=2, label='5G first policy'),
            Line2D([0], [0], color='c', lw=2, label='Priority-based policy'),
            Line2D([0], [0], color='b', lw=2, label='Random policy'),
        ]
        plt.legend(handles=legend_elements, loc=2)

        plt.grid(True)
        plt.xlabel('Average events per second')
        plt.ylabel(r'$\gamma$' + ' (prioritized bits)')
        plt.title(r'$\gamma$ vs $\lambda$ for different policies')
        plt.savefig('results_' + root_folder.split('/')[0] + '_offperiod.png', dpi=600)
        plt.show()
        exit()
    else:
        dataframe = pd.read_csv("data_first100k.csv", header=None, names=["node_id", "timestamp", "value_hrf"])
        dataframe["timestamp"] = dataframe["timestamp"].apply(timestamp_to_datetime)

        offsets = get_offsets(dataframe, timedelta(days=1, hours=0, minutes=0, seconds=0))

        foo()

        threshold = 0.2 # change bash here
        print("Loading dataframe")
        dataframe = dataframe[dataframe.iloc[:, 2] > threshold]
        # First, apply threshold, then compute thresholds
        offsets = get_offsets(dataframe, timedelta(days=1, hours=0, minutes=0, seconds=0))

        print("Slicing data frame") # 25.453326425502084
        dataframe = slice_df_in_days(dataframe, offsets)
        print(1/np.mean([len(v)/(3600*24) for v in dataframe]))
        exit()
        MAX_ITERATION = 1000
        print("Running training phase")
        test_solver(cmaes)
        # test_solver(pepg)
        # test_solver(openes)