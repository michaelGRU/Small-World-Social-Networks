import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

path, cluster = [], []
sample = 10
num_of_probability = 5
list_of_ones = [1] * num_of_probability
p_rewire = []
for i in range(num_of_probability):
    list_of_ones[i] = list_of_ones[i] * (0.1 ** i)
    p_rewire.append(list_of_ones[i])
p_rewire = p_rewire[::-1]
for p in p_rewire:
    path_i, cluster_i = [], []
    for n in range(sample):
        G = nx.watts_strogatz_graph(300, 10, p)
        path_i.append(nx.average_shortest_path_length(G))
        cluster_i.append(nx.average_clustering(G))
    path.append(sum(path_i) / len(path_i))
    cluster.append(sum(cluster_i) / len(cluster_i))

plt.scatter(p_rewire, [x / path[0] for x in path], label="APL(p) / APL(0)")
plt.scatter(p_rewire, [x / cluster[0] for x in cluster], label="CC(p) / CC(0)")
plt.plot(p_rewire, [x / path[0] for x in path])
plt.plot(p_rewire, [x / cluster[0] for x in cluster])
ax = plt.gca()
ax.set_xscale("log")
plt.xlabel("Probability of Re-wiring")
plt.ylabel("CC and APL Ratio")
plt.title("The Small World Phenomena")
plt.legend()
plt.show()
