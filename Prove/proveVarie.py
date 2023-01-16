import networkx as nx
import matplotlib.pyplot as plt

g = nx.Graph()

g.add_edge('a', 'b', weight=0.1)
g.add_edge('b', 'c', weight=1.5)
g.add_edge('a', 'c', weight=1.0)
g.add_edge('c', 'd', weight=2.2)
g.add_edge('c', 'e', weight=1.2)
g.add_edge('c', 'f', weight=2.0)
g.add_edge('e', 'f', weight=0.2)
g.add_edge('a', 'e', weight=2.2)

print(nx.shortest_path(g, 'b', 'd'))
print(nx.shortest_path(g, 'b', 'd', weight='weight'))

g.add_node('a', color='red')
print("i nodi della rete sono:")
print(list(g.nodes))

print("i vicini di A sono:")
print(g.adj['a'])
print("quindi A ha grado:")
print(g.degree['a'])

for n1, n2, attr in g.edges(data=True):
    print (n1, n2, attr['weight'])

print(g.nodes.data())

pos = nx.spring_layout(g, seed=3113794652)
options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
nx.draw_networkx_nodes(g, pos, nodelist=list(g.nodes), node_color="tab:red", **options)
edlabels = nx.get_edge_attributes(g,'weight')
nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
nx.draw_networkx_edge_labels(g,pos,edge_labels=edlabels)
nx.draw_networkx_labels(g, pos, font_size=22, font_color="whitesmoke")
plt.tight_layout()
plt.axis("off")
plt.show()