from matplotlib import pyplot as plt
from numpy import array
from matplotlib.lines import Line2D

num_samples = 8
width = (1 / num_samples) * 0.9  # num_samples lambdas diferentes
separation = (1 / num_samples) * 0.1
ind = [0, 1.5, 3, 4.5]  # 3 policies

"""
[13688539.375104211, 13074189.604273858, 11890094.028525557, 11754003.831654854, 11475045.956471901, 10445479.314985123, 9134988.682991225, 8313706.229168314, 8045508.763141763]
[7957621.208641222, 9089021.773355214, 9295720.747515237, 9333628.501689592, 9503942.560254838, 9338886.84843388, 8377482.665854041, 8177393.927022562, 8060125.150579404]
[8805551.114829995, 10506190.408570165, 10062910.01587345, 9733556.465978472, 9455171.869794918, 8675042.648292972, 7751232.26211008, 6941776.20608854, 6346474.4889248]
[7959592.616503206, 8919711.979416043, 8916576.643440917, 8849528.89488814, 8789800.301059721, 7725642.090550985, 7274545.93054969, 6734124.795271104, 6088859.165669142]
[array([0.34319288, 0.03729086, 0.61951626]), array([0.55636432, 0.09314731, 0.35048837]), array([0.75760932, 0.18078835, 0.06160233]), array([0.72733334, 0.20394499, 0.06872167]), array([0.63904391, 0.2913246 , 0.06963149]), array([0.47635509, 0.4763121 , 0.04733281]), array([0.350089 , 0.5836667, 0.0662443]), array([0.29514064, 0.62395233, 0.08090702]), array([0.25442798, 0.72914475, 0.01642727])]
[array([0.21221146, 0.02453431, 0.76325423]), array([0.49589493, 0.05855862, 0.44554645]), array([0.64173279, 0.11134165, 0.24692556]), array([0.65577876, 0.13895844, 0.20526281]), array([0.61837436, 0.24911145, 0.13251419]), array([0.45265752, 0.46074179, 0.08660069]), array([0.37413146, 0.56279051, 0.06307803]), array([0.3056067 , 0.64476352, 0.04962978]), array([0.23658659, 0.73273697, 0.03067644])]
[array([0.21237046, 0.02438791, 0.76324163]), array([0.4987911 , 0.05559094, 0.44561796]), array([0.67520127, 0.07792905, 0.24686968]), array([0.71196527, 0.08282991, 0.20520481]), array([0.75752934, 0.10990409, 0.13256656]), array([0.74779351, 0.16566764, 0.08653884]), array([0.7177827 , 0.21914635, 0.06307095]), array([0.68650413, 0.26387455, 0.04962132]), array([0.70071538, 0.26859738, 0.03068724])]
[array([0.21238373, 0.02455308, 0.76306319]), array([0.41355577, 0.05873961, 0.52770462]), array([0.46357544, 0.11139601, 0.42502855]), array([0.46916456, 0.13809113, 0.39274431]), array([0.47931945, 0.22606896, 0.29461159]), array([0.48797491, 0.30065516, 0.21136993]), array([0.48959043, 0.35502142, 0.15538815]), array([0.4962136 , 0.40157216, 0.10221424]), array([0.49777011, 0.4167368 , 0.08549309])]
"""
usages_ann = [array([0.34319288, 0.03729086, 0.61951626]) * 100,
              array([0.55636432, 0.09314731, 0.35048837]) * 100,
              array([0.75760932, 0.18078835, 0.06160233]) * 100,
              array([0.72733334, 0.20394499, 0.06872167]) * 100,
              array([0.63904391, 0.2913246, 0.06963149]) * 100,
              array([0.47635509, 0.4763121, 0.04733281]) * 100,
              array([0.350089, 0.5836667, 0.0662443]) * 100,
              array([0.29514064, 0.62395233, 0.08090702]) * 100,
              array([0.25442798, 0.72914475, 0.01642727])* 100]

usages_5gf = [array([0.21221146, 0.02453431, 0.76325423]) * 100,
              array([0.49589493, 0.05855862, 0.44554645]) * 100,
              array([0.64173279, 0.11134165, 0.24692556]) * 100,
              array([0.65577876, 0.13895844, 0.20526281]) * 100,
              array([0.61837436, 0.24911145, 0.13251419]) * 100,
              array([0.45265752, 0.46074179, 0.08660069]) * 100,
              array([0.37413146, 0.56279051, 0.06307803]) * 100,
              array([0.3056067, 0.64476352, 0.04962978]) * 100,
              array([0.23658659, 0.73273697, 0.03067644])* 100]

usages_pri = [array([0.21237046, 0.02438791, 0.76324163]) * 100,
              array([0.4987911, 0.05559094, 0.44561796]) * 100,
              array([0.67520127, 0.07792905, 0.24686968]) * 100,
              array([0.71196527, 0.08282991, 0.20520481]) * 100,
              array([0.75752934, 0.10990409, 0.13256656]) * 100,
              array([0.74779351, 0.16566764, 0.08653884]) * 100,
              array([0.7177827, 0.21914635, 0.06307095]) * 100,
              array([0.68650413, 0.26387455, 0.04962132]) * 100,
              array([0.70071538, 0.26859738, 0.03068724])* 100]

usages_rnd = [array([0.21238373, 0.02455308, 0.76306319]) * 100,
              array([0.41355577, 0.05873961, 0.52770462]) * 100,
              array([0.46357544, 0.11139601, 0.42502855]) * 100,
              array([0.46916456, 0.13809113, 0.39274431]) * 100,
              array([0.47931945, 0.22606896, 0.29461159]) * 100,
              array([0.48797491, 0.30065516, 0.21136993]) * 100,
              array([0.48959043, 0.35502142, 0.15538815]) * 100,
              array([0.4962136, 0.40157216, 0.10221424]) * 100,
              array([0.49777011, 0.4167368, 0.08549309])* 100]

usages_ann = usages_ann[::-1]

usages_5gf = usages_5gf[::-1]

usages_pri = usages_pri[::-1]

usages_rnd = usages_rnd[::-1]


custom_lines = [Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)]

# usages_ann = [np.array([0., 1., 0.]), np.array([0., 1., 0.]), np.array([0., 1., 0.]),
#               np.array([0., 0.99856311, 0.00143689]), np.array([0.20574333, 0.78846021, 0.00579646]),
#               np.array([0.2326505, 0.75865313, 0.00869637]), np.array([0.22838645, 0.74750454, 0.02410901])][::-1]
#
#
# usages_5gf = [np.array([0., 1., 0.]), np.array([0., 1., 0.]), np.array([0., 1., 0.]),
#               np.array([0.00146373, 0.99853627, 0.]), np.array([0.20660821, 0.79339179, 0.]),
#               np.array([0.23268349, 0.76731651, 0.]), np.array([0.23259202, 0.76740798, 0.])][::-1]
#
# usages_pri = [np.array([0.50052328, 0.49947672, 0.]), np.array([0.49948024, 0.50051976, 0.]),
#               np.array([0.49978189, 0.50021811, 0.]), np.array([0.49996975, 0.50003025, 0.]),
#               np.array([0.4994492, 0.5005508, 0.]), np.array([0.50038051, 0.49961949, 0.]),
#               np.array([0.50024494, 0.49975506, 0.])][::-1]
#
# usages_rnd = [np.array([0.50033074, 0.49966926, 0.]), np.array([0.50113527, 0.49886473, 0.]),
#               np.array([0.50059005, 0.49940995, 0.]), np.array([0.50012053, 0.49987947, 0.]),
#               np.array([0.50008663, 0.49991337, 0.]), np.array([0.49957326, 0.50042674, 0.]),
#               np.array([0.5003174, 0.4996826, 0.])][::-1]


fig, ax = plt.subplots()

for i in range(num_samples):
    ax.bar(ind[0] + i * width + separation * i, usages_ann[i][0], width, color='r')
    ax.bar(ind[0] + i * width + separation * i, usages_ann[i][1], width, bottom=usages_ann[i][0], color='b')
    ax.bar(ind[0] + i * width + separation * i, usages_ann[i][2], width, bottom=usages_ann[i][0]+usages_ann[i][1], color='k')

for i in range(num_samples):
    ax.bar(ind[1] + i * width + separation * i, usages_5gf[i][0], width, color='r')
    ax.bar(ind[1] + i * width + separation * i, usages_5gf[i][1], width, bottom=usages_5gf[i][0], color='b')
    ax.bar(ind[1] + i * width + separation * i, usages_5gf[i][2], width, bottom=usages_5gf[i][0]+usages_5gf[i][1], color='k')

for i in range(num_samples):
    ax.bar(ind[2] + i * width + separation * i, usages_pri[i][0], width, color='r')
    ax.bar(ind[2] + i * width + separation * i, usages_pri[i][1], width, bottom=usages_pri[i][0], color='b')
    ax.bar(ind[2] + i * width + separation * i, usages_pri[i][2], width, bottom=usages_pri[i][0]+usages_pri[i][1], color='k')

for i in range(num_samples):
    ax.bar(ind[3] + i * width + separation * i, usages_rnd[i][0], width, color='r')
    ax.bar(ind[3] + i * width + separation * i, usages_rnd[i][1], width, bottom=usages_rnd[i][0], color='b')
    ax.bar(ind[3] + i * width + separation * i, usages_rnd[i][2], width, bottom=usages_rnd[i][0]+usages_rnd[i][1], color='k')


ax.legend(custom_lines, ['% of packets sent with LoRa', '% of packets sent with 5G', '% of packets dropped']) #, loc="upper center", bbox_to_anchor=(0.5, 1.22))

plt.text(-0.2, -9, "Proposed policy", fontsize=10)
plt.text(1.35, -9, "5G-first policy", fontsize=10)
plt.text(3.45, -13.2, "Priority-based\npolicy", fontsize=10, horizontalalignment='center')
plt.text(4.35, -9, "Random policy", fontsize=10)
ax.axes.get_xaxis().set_visible(False)
plt.show()
# lora = plt.bar(ind, menMeans, width, color='#d62728')
# fiveg = plt.bar(ind, womenMeans, width, bottom=menMeans, yerr=womenStd)
# drop = plt.bar(ind, womenMeans, width, color='#000000', bottom=menMeans, yerr=womenStd)
