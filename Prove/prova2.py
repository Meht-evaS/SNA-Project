import networkx as nx
import random
import matplotlib.pyplot as plt

I = nx.fast_gnp_random_graph(20, 50, seed=None, directed=False)
for x in range(I.number_of_nodes()):
  I.add_nodes_from([(x, {"state": "susceptible", "color": "blue", "ninfected": 0})])

pos = nx.spring_layout(I, seed=3113794652)
options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
nx.draw_networkx_nodes(I, pos, nodelist=list(I.nodes), node_color="tab:red", **options)
edlabels = nx.get_edge_attributes(I,'weight')
nx.draw_networkx_edges(I, pos, width=1.0, alpha=0.5)
nx.draw_networkx_edge_labels(I,pos,edge_labels=edlabels)
nx.draw_networkx_labels(I, pos, font_size=22, font_color="whitesmoke")

plt.tight_layout()
plt.axis("off")
plt.show()