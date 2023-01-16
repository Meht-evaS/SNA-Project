import networkx as nx
import random

I = nx.fast_gnp_random_graph(15, 40, seed=None, directed=False)
for x in range(I.number_of_nodes()):
  I.add_nodes_from([(x, {'state': 'susceptible', 'color': 'blue', 'ninfected': 0})])

#valori iniziali
pinitinf = 0.30
ptrans = 0.10
trec = 7
tsus = 15
cicles = 4

rstato = nx.get_node_attributes(I, "state")
rtrecleft = nx.get_node_attributes(I, "trecleft")
rtimmunity = nx.get_node_attributes(I, "timmunity")
rninfected = nx.get_node_attributes(I, "timmunity")

#funzione per infettare percentuale iniziale dei nodi:
#prende come parametro percentuale, viene moltiplicata per il numero di nodi della rete e arrotondato
#con un while, prima verifichiamo di non aver già infettato il nodo, quindi lo infettiamo e dimuniamo
#numero di nodi ancora da infettare
def infettainit(p):
    initinf = round(p*(I.number_of_nodes()),0)
    while initinf > 0:
        x = random.randint(0,I.number_of_nodes()-1)
        if rstato[x]!='infected':
            I.add_node(x, state='infected')
            initinf -= 1

infettainit(pinitinf)
print(I.nodes.data())

for e in range(cicles):
    for nodo in (I.nodes):
        if rstato[nodo]=='infected':
            for vicino in (I.adj[nodo]):
                if rstato[vicino]=='susceptible' and round(random.uniform(0.00, 1.00), 2) <= ptrans:
                    #se un nodo è infettato, prendiamo tutti i vicini infettabili (suscettibili)
                    #prendiamo la probabilità di infezione e un numero random tra 0 e 1
                    #se il numero è minore della probabilità di infezione, il vicino viene contagiato
                    #quindi gli assegniamo il tempo di recupero e aumentiamo il count dei nodi infettati dal nodo
                    I.add_node(vicino, state='infected')
                    I.add_node(vicino, trecleft=trec + 1)
                    I.add_node(nodo, ninfected=rninfected[nodo] + 1)
    if rstato[nodo] == 'infected':
        I.add_node(nodo, color='red') #coloriamo di rosso i nodi infetti
        if rtrecleft[nodo] >= 1:
            I.add_node(nodo, timmunity=rtrecleft[nodo]-1)
            if rtrecleft[nodo] == 0:
                I.add_node(nodo, state = 'recovered')
                I.add_node(nodo, timmunity = tsus + 1)
                I[nodo]['timmunity'] = tsus + 1
    if rstato[nodo] == 'recovered':
        I.add_node(nodo, color='green') #coloriamo di verde i nodi guariti
        if rtimmunity[nodo] >= 1:
            I.add_node(nodo, timmunity=rtimmunity[nodo]-1)
            if rtimmunity[nodo] == 0:
                I.add_node(nodo, state='susceptible')
                I.add_node(nodo, color='blue') #riportiamo al colore iniziale di blu i nodi di nuovo suscettibili

print("dopo esecuzione")
print(I.nodes.data())              

