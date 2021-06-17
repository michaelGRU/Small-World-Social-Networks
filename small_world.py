# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Watts-Strogatz Model, Small-World Networks, and Graph Theory
#
# In the 1960s, social psychologist Stanley Milgram(the same dude who asked participants to administer shocks to people) conducted a series of experiments to investigate the probability that two random people would know each other via mutual connections. The findings suggested that the average path length for social networks of people is around six, indicating that the structure of human society resembles that of a small-world network. In the 1990s, two mathematicians published a model known as the Watts-Strogatz network, supporting that real-world networks, such as film actors network, power grid, and neural network, exhibit small-world property. In essence, many seemingly large networks can be spanned by relatively small path lengths between two random nodes.
#
# It's an interesting phenomenon that real world social networks tend to have high average clustering but smaller diameters(short paths on a global scale). We can connect two people via a surprising short chain of friendships.
#
# We will first take a look at the properties of a word co-occurrence network I created analyzing the James Bond novel From Russia with Love. The source code can be found here: https://github.com/michaelGRU/Co-occurrence-Networks/blob/main/co_occurrence_network.py

# %%
import my_word_co_network

# %% [markdown]
# You can find the explanation of this network on my Github page. For now, we will only look at the graph theory aspects of this network. The network we generated has a high average clustering coefficient, and a low average degrees of separation. Watts and Strogatz define a small-world network as "average path length is close to the average path length for a random network of the same size, but clustering coefficient is much bigger than that of a random network of the same size." In other words, the James Bond network we generated has the small-world property.
#
# ## Graph Theory Notation
#
# Let G denotes an object that consists of a collection of nodes V and edges or arcs E. We can express G as the following:
# $$G = (V,E)$$
#
# Suppose there is a directed connection between two nodes, $v_1$ and $v_2$. We denote it as  $$\left(v_{1}, v_{2}\right)$$
# This is referred to as an arc.
#
# Suppose the connection is undirected, we denoite it as $$\left(v_{1}: v_{2}\right)$$
# This is referred to as an edge.
#
# The neighbours of a node $v_i$ is denoted as $N(v_i)$.
#
# The shortest length between two nodes is denoted as $d\left(v_{i}, v_{j}\right)$. This is also known as the geodesic paths.
#
# The longest distance between two nodes is denoted as $\max _{v_{i}, v_{j} \in V} d\left(v_{i}, v_{j}\right)$ or the diameter of the graph (diam(G)).
#
# The distance is $\infty$ is the nodes are not connected.
# %% [markdown]
# The average path length of a connected graph G is
#
# $$L(G)=\frac{1}{n(n-1)} \sum_{u, v \in V} d(u, v)$$
#
# The Clustering Coefficient of a graph G is the average of the clustering coefficients of its nodes is
#
# $$CC(G)=\frac{1}{V} \sum_{v \in V} c c(v)$$
#
# The Clustering Coefficient CC(v) is calculated using the number of pairs of adjacent neighbors divided by the number of pairs of neighbors.
#
# CC(v) Example:
#
# In the following un-directed network, each node represents a person, and the edges represent friendship.

# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G_fren = nx.Graph()
G_fren.add_edges_from(
    [
        ("Chloe", "Victoria"),
        ("Chloe", "Jackson"),
        ("Jackson", "Victoria"),
        ("Chloe", "Kate"),
        ("Rubert", "Chloe"),
        ("Rubert", "Victoria"),
    ]
)
nx.draw_networkx(G_fren)
plt.gca().margins(0.1, 0.1)


# %%
# print the pairs of nodes
print(G_fren.edges)

# %% [markdown]
# We can see that Jackson is friends with Chloe. Notice that the direction does not matter here. If Jackson is friends with Chloe, Chloe is friends with Jackson by default. Chloe is also friends with Kate, but Jackson and Kate are not directly connected. We can represent this graph using an adjacency matrix.

# %%
adj_mat = nx.to_numpy_matrix(G_fren)
print(adj_mat)

# %% [markdown]
# The main diagonal (diagonal from the top left corner to the bottom right corner of the matrix) is 0 since, in our model, you can only be friends with people other than yourself.
#
# The Clustering Coefficient for Chloe is 0.33 since the number of pairs of adjacent neighbors is 2, and the number of pairs of neighbors is 6.

# %%
nx.clustering(G_fren)


# %%
nx.average_clustering(G_fren)

# %% [markdown]
# ## Real-World Network
# When it comes to real-world networks, typically we would expect a large number of nodes and a much smaller number of neighbors compared to the number of nodes. In layman's terms, for instance, each node can represent a person. Each person is connected to their friends. There are around 8 billion people in the world, and the number of friends a person has is much less than this number. Further, there is a degree of randomness in the real-world network, such that it's not realistic to expect the network to be perfectly symmetric. On top of that, real-world networks tend to exhibit the small-world property(ie. small average shortest path length and high clustering).
#
#
#
#
#
#
#
#
#
#
# %% [markdown]
# We will look at some other networks that don't have the small-world property.
# ## Ring Network (Average Distance: O(n))
# Characteristics:
# - Symmetric
# - High average clustering
# - High average shortest path length

# %%
# each node is connected to k nearest neighbors in ring topology
k = 4
# the number of nodes
n = 12
# the probability of rewiring each edge
p = 0
# seed for random number generator (default=None)
seed = None

# generate the ring network
G_ring = nx.watts_strogatz_graph(n, k, p, seed=None)
pos = nx.circular_layout(G_ring)
nx.draw_networkx_edges(G_ring, pos, width=2, alpha=0.7)
nx.draw_networkx_nodes(G_ring, pos, node_shape="h", node_color="k")

# %% [markdown]
# Notice that when we scale up the above network, it exhibits a high average clustering and a high average distance between vertices, as demonstrated below. We scale the number of nodes to 500, and its neighbors k to 10.

# %%
n_, k_ = 500, 10
G_ring_ = nx.watts_strogatz_graph(n_, k_, p)

# %% [markdown]
# The average degrees of separation is:

# %%
round(nx.average_shortest_path_length(G_ring_), 2)

# %% [markdown]
# The average clustering is:

# %%
round(nx.average_clustering(G_ring_), 2)

# %% [markdown]
# Notice that the average degrees of separation is significantly larger than the six degrees of separation we discussed previously in Milgram's experiment. The ring network is not an accurate representation of a small world network.
# ## Random Network (Average Distance: O(log n))
# Characteristics:
# - Asymmetric
# - Low average clustering
# - Low average degrees of separation
#
# We can mimic a random network by scaling up the probability of rewiring each edge.

# %%
G_ring_random = nx.watts_strogatz_graph(n, k, 1)
pos = nx.circular_layout(G_ring_random)
nx.draw_networkx_edges(G_ring_random, pos, width=2, alpha=0.9)
nx.draw_networkx_nodes(G_ring_random, pos, node_shape="h", node_color="k")

# %% [markdown]
# We can again look at the average shortest path length and average clustering when n is sufficiently large. Note that the higher the p value is, the more randomness is introduced to the graph.

# %%
G_ring_random_ = nx.watts_strogatz_graph(3000, 10, 1)
round(nx.average_shortest_path_length(G_ring_random_), 2)


# %%
round(nx.average_clustering(G_ring_random_), 2)

# %% [markdown]
# Notice that this is not a good representation of a small world network either. The average shortest path length is a lot closer to 6. However, the average clustering is 0.
# %% [markdown]
# ## Watts-Strogatz Model
# Let APL(p) and ACC(p) be the average shortest path length and clustering coefficient for a ring lattice with rewiring. The probability of rewiring is p. (0 <= p <= 1).
#
# Watts and Strogatz studied $\frac{APL(p)}{APL(0)}$ and $\frac{ACC(p)}{ACC(0)}$. Both functions start at 1 and end close to 0 as functions of p over [0, 1]. We can plot the two variables.

# %%
# find the average shortest path length and average clustering for different probabilities of rewiring
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
# plot p vs APL(p)/APL(0) and p vs CC(p)/CC(0) on the same plot
plt.scatter(p_rewire, [apl / path[0] for apl in path], label="APL(p) / APL(0)")
plt.scatter(p_rewire, [cc / cluster[0] for cc in cluster], label="CC(p) / CC(0)")
plt.plot(p_rewire, [x / path[0] for x in path])
plt.plot(p_rewire, [x / cluster[0] for x in cluster])
ax = plt.gca()
ax.set_xscale("log")
plt.xlabel("Probability of Re-wiring")
plt.ylabel("CC and APL Ratio")
plt.title("The Small World Phenomena")
plt.legend()

# %% [markdown]
# We can see that APL(p)/APL(0) and CC(p)/CC(0) behave quite differently. For average clustering, the dropping rate is much slower than the average path length. APL ratio looks concave up, whereas the CC ratio looks concave down. When the probability of rewiring increase by a small amount, it has a dramatic impact on the average shortest path length without significantly changing the average clustering coefficient. Graphs that are produced in the middle region are highly clustered but have a short average distance between nodes, exhibiting the small world property.

# %%
from IPython.display import Image
from IPython.core.display import HTML

Image(
    url="https://www.researchgate.net/profile/Olaf-Sporns/publication/50268221/figure/fig2/AS:203034706616323@1425418654961/The-Watts-Strogatz-model-of-the-small-world-The-network-at-the-upper-left-hand-corner.png"
)

# %% [markdown]
# | Network| Average Path Length | Clustering |
# | --- | --- | --- |
# | Regular Ring | High | High |
# | Small World | Low | High |
# | Random | Low | Low |
# %% [markdown]
# **References**
#
# Sporns, Olaf. (2011). The Non-Random Brain: Efficiency, Economy, and Complex Dynamics
#
# Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of ‘small-world’
# networks. Nature, 393(6684).
#
# Platt, Edward, (2019). Network Science with Python and NetworkX Quick Start Guide.
# %% [markdown]
#
