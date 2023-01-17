import networkx as nx
import random
import csv
import sys

infected_nodes = []


I = nx.fast_gnp_random_graph(15, 20, seed=None, directed=False)
for x in range(I.number_of_nodes()):
  I.add_nodes_from([(x, {'state': 'susceptible', 'color': 'blue', 'count_infected': 0})])

#valori iniziali
p_init = 0.20
p_trans = 0.05
t_rec = 7
t_sus = 15
t_step = 20
iterations = 4

def read_state(node):
    return nx.get_node_attributes(I, "state")[node]

def read_recovery_time_left(node):
    return nx.get_node_attributes(I, "recovery_time_left")[node]

def read_immunity_time_left(node):
    return nx.get_node_attributes(I, "immunity_time_left")[node]

def read_count_infected(node):
    return nx.get_node_attributes(I, "count_infected")[node]

#funzione per infettare percentuale iniziale dei nodi:
#prende come parametro percentuale, viene moltiplicata per il numero di nodi della rete e arrotondato
#con un while, prima verifichiamo di non aver già infettato il node, quindi lo infettiamo e dimuniamo
#numero di nodi ancora da infettare
def infettainit(p):
    initinf = round(p*(I.number_of_nodes()),0)
    while initinf > 0:
        x = random.randint(0,I.number_of_nodes()-1)
        if read_state(x)!='infected':
            I.add_node(x, state='infected')
            I.add_node(x, color='red')
            I.add_node(x, recovery_time_left=t_rec)
            initinf -= 1
            infected_nodes.append(x)

infettainit(p_init)
for element in I.nodes.data() :
    print(element)
print("\n\n------------------------------------------------------------------------------------\n\n")


#for e in range(iterations):
for step in range(t_step):
    
    new_infected_nodes = []
    print(infected_nodes)
    copy_infected_nodes = infected_nodes.copy()

    for node in (infected_nodes):
        print("Sto lavorando con il nodo " + str(node))
        print(infected_nodes)

        for neighbor in (I.adj[node]):
            if read_state(neighbor)=='susceptible' and round(random.uniform(0.00, 1.00), 2) <= p_trans:
                #se un node è infettato, prendiamo tutti i vicini infettabili (suscettibili)
                #prendiamo la probabilità di infezione e un numero random tra 0 e 1
                #se il numero è minore della probabilità di infezione, il neighbor viene contagiato
                #quindi gli assegniamo il tempo di recupero e aumentiamo il count dei nodi infettati dal node
                I.add_node(neighbor, state='infected')
                I.add_node(neighbor, recovery_time_left=(t_rec))
                I.add_node(neighbor, color='red')
                I.add_node(node, count_infected=(read_count_infected(node) + 1))
                new_infected_nodes.append(neighbor)
                
        if read_recovery_time_left(node) >= 1:
            I.add_node(node, recovery_time_left=(read_recovery_time_left(node)-1))
            if read_recovery_time_left(node) == 0:
                I.add_node(node, state = 'recovered')
                I.add_node(node, immunity_time_left = (t_sus + 1))
                copy_infected_nodes.remove(node)

    for node in (I.nodes):
        if read_state(node) == 'recovered':
            I.add_node(node, color='green') #coloriamo di verde i nodi guariti
            if read_immunity_time_left(node) >= 1:
                I.add_node(node, immunity_time_left=(read_immunity_time_left(node)-1))
                if read_immunity_time_left(node) == 0:
                    I.add_node(node, state='susceptible')
                    I.add_node(node, color='blue') #riportiamo al colore iniziale di blu i nodi di nuovo suscettibili    
    
    for element in I.nodes.data() :
        print(element)
    print("\n\n")

    infected_nodes = copy_infected_nodes + new_infected_nodes