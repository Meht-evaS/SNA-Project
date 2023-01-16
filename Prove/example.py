import networkx as nx
import random

def simulate_virus_transmission(G, transmission_rate):
    infected_nodes = set() # set to store infected nodes
    for node in G.nodes():
        if random.random() < transmission_rate: # randomly infecting some nodes
            infected_nodes.add(node)
    for infected_node in infected_nodes:
        for neighbor in G[infected_node]:
            if random.random() < transmission_rate: # infecting neighbors with probability of transmission
                infected_nodes.add(neighbor)
    return infected_nodes

# Example usage:
G = nx.erdos_renyi_graph(100, 0.1) # generate a random graph with 100 nodes and edge probability 0.1
transmission_prob = 0.05
final_infected = simulate_virus_transmission(G, transmission_prob)
print(f'Number of infected nodes: {len(final_infected)}')
