import networkx as nx
import random
import csv
import sys

# Rendering curve tramite interpolazione 
import numpy as np
from scipy.interpolate import make_interp_spline # pip install scipy
import matplotlib.pyplot as plt

from math import log

from operator import itemgetter


#################################################################
#                                                               #
#            VARIABILI GLOBALI E SETTING ENVIRONMENT            #
#                                                               #
#################################################################

I = nx.Graph()

header = ['source','target']
infected_nodes = []
statistics_graph = []
max_spreader = []

errore_lettura_csv = 'Il file CSV passato in input deve iniziare con il seguente header:\t\t' + header[0] + ',' + header[1] + ' [, ...]\n'

#grafo_csv = input("Inserisci il nome del file csv da leggere: ")

#valori iniziali
#grafo_csv = 'Prove_mod_cris\graph.csv'
grafo_csv = 'graph3.csv'
p_init = 0.05
p_trans = 0.10
t_rec = 3
t_sus = 7
t_step = 20
iterations = 4




#################################################################
#                                                               #
#                FUNZIONI CONTROLLO CSV DI INPUT                #
#                                                               #
#################################################################

def test_header(csv_header):
    if (csv_header[0] != header[0] or csv_header[1] != header[1]):
        errore = "Errore lettura file '" + grafo_csv + "': header non corretto!\n\n"
        sys.exit(errore + errore_lettura_csv)

def test_len(len_row, num_row):
    if (len_row < 2): # 2 = len(header)
        errore = "Errore lettura file '" + grafo_csv + "': alla riga " + str(num_row) + " mancano dei parametri indispensabili!\n\n"
        sys.exit(errore)

def test_and_add_edge(row, num_row, warnings):
    try:
        test_len(len(row), num_row)
        I.add_edge(int(row[0]), int(row[1]))
        #print("Aggiunto arco riga " + str(num_row) + ", da nodo " + row[0] + " a nodo " + row[1])
    except ValueError as ve:
        errore = "Riga " + str(num_row) + " del file '" + grafo_csv + "': l'ID del nodo '" + header[0] + "' o '" + header[1] + "' non è un numero intero!"
        warnings.append(errore)
    except Exception as ex:
        print("Si è verificato un errore imprevisto durante l'aggiunta degli archi al grafo: " + str(ex))
    finally:
        return warnings

#################################################################
#                                                               #
#                FUNZIONI LETTURA ATTRIBUTI NODI                #
#                                                               #
#################################################################

def read_state(node):
    return nx.get_node_attributes(I, "state")[node]

def read_recovery_time_left(node):
    return nx.get_node_attributes(I, "recovery_time_left")[node]

def read_immunity_time_left(node):
    return nx.get_node_attributes(I, "immunity_time_left")[node]

def read_count_infected(node):
    return nx.get_node_attributes(I, "count_infected")[node]

def read_temporary_count_infected(node):
    return nx.get_node_attributes(I, "temporary_count_infected")[node]

def read_neighbor_infected(node):
    return nx.get_node_attributes(I, "neighbor_infected")[node]

def print_stats(sus, inf, rec):
    print('Susceptible: ' + str(sus))
    print('Infected: ' + str(inf))
    print('Recovered: ' + str(rec))


#################################################################
#                                                               #
#                  FUNZIONE INFEZIONE INIZIALE                  #
#                                                               #
#################################################################

#funzione per infettare percentuale iniziale dei nodi:
#prende come parametro percentuale, viene moltiplicata per il numero di nodi della rete e arrotondato.
#con un while, prima verifichiamo di non aver già infettato il node, quindi lo infettiamo e dimuniamo il
#numero di nodi ancora da infettare
def infettainit(p):

    '''print('1 -- I.number_of_nodes(): ' + str(I.number_of_nodes())) 
    print(I)
    print('Edge list:\n' + str(list(I.edges)) + '\n')
    print('Node list:\n' + str(list(I.nodes)) + '\n')'''
           
    initinf = int(round(p * (I.number_of_nodes()), 0))
    print('Susceptible: ' + str(I.number_of_nodes()-initinf))
    print('Infected: ' + str(initinf))
    print('Recovered: 0')
    st_tuple = (I.number_of_nodes()-initinf, initinf, 0)
    statistics_graph.append(st_tuple)
    while initinf > 0:
        #print('2 -- I.number_of_nodes(): ' + str(I.number_of_nodes()))
        x = random.choice(list(I.nodes))
        #print('x: ' + str(x) + '\n\n')

        if read_state(x) != 'infected':
            I.add_node(x, state='infected')
            I.add_node(x, color='red')
            I.add_node(x, recovery_time_left=t_rec)
            initinf -= 1
            infected_nodes.append(x) 





#################################################################
#                                                               #
#                INIZIO PROGRAMMA - LETTURA CSV                 #
#                                                               #
#################################################################


with open(grafo_csv, encoding='utf8') as csv_file:
    warnings = []

    csv_reader = csv.reader(csv_file, delimiter=',')

    csv_header = next(csv_reader)
    print(str(csv_header) + '\n\n')

    test_header(csv_header) # Controllo se header CSV corretto, altrimenti interruzione programma

    csv_type = next(csv_reader) # Lo uso giusto per creare un WARNING in caso contenga un grafo diretto
    print('\n\n' + str(csv_type))

    if (len(csv_type) > 2 and csv_type[2] == 'directed'):
        warnings.append("Il file '" + grafo_csv + "' contiene un grafo 'diretto' ma verrà trasformato in un grafo 'non diretto'") 

    warnings = test_and_add_edge(csv_type, 2, warnings) # Aggiungo l'arco della riga appena letta, sennò lo perderei

    num_row = 2
    for row in csv_reader:
        num_row += 1
        print(row)

        warnings = test_and_add_edge(row, num_row, warnings) 


    I.remove_edges_from(nx.selfloop_edges(I)) # rimuovo i self loop

    if (len(warnings) > 0):
        print('\n\nWARNING:')
        for warning in warnings:
            print(warning)
        
        print('\n\n')


print('\n\n')
print('Edge list:\n' + str(list(I.edges)) + '\n')
print('Node list:\n' + str(list(I.nodes)) + '\n')
print('Number of nodes:\n' + str(I.number_of_nodes()) + '\n')



#################################################################
#                                                               #
#                CORPO PROGRAMMA - STEP CONTAGIO                #
#                                                               #
#################################################################

# I = nx.fast_gnp_random_graph(15, 20, seed=None, directed=False)

for node in (I.nodes): 
    '''lst = []
    for i in range(t_step):
        lst.append([])'''

    #I.add_nodes_from([(node, {'state': 'susceptible', 'color': 'blue', 'count_infected': 0, 'neighbor_infected': lst})])
    I.add_nodes_from([(node, {'state': 'susceptible', 'color': 'blue', 'count_infected': 0, 'temporary_count_infected': 0})])

    


infettainit(p_init)

'''for element in I.nodes.data() :
    print(element)'''

print("\n\n------------------------------------------------------------------------------------\n\n")

#sys.exit()

#for e in range(iterations):
for step in range(t_step):
    print(str(step))
    new_infected_nodes = []
    #print(infected_nodes)
    random.shuffle(infected_nodes)
    #print(infected_nodes)
    copy_infected_nodes = infected_nodes.copy()

    for node in (infected_nodes):
        #print("Sto lavorando con il nodo " + str(node))
        #print(infected_nodes)

        for neighbor in (I.adj[node]):
            if read_state(neighbor) =='susceptible' and round(random.uniform(0.00, 1.00), 2) <= p_trans:
                #se un node è infettato, prendiamo tutti i vicini infettabili (suscettibili)
                #prendiamo la probabilità di infezione e un numero random tra 0 e 1
                #se il numero è minore della probabilità di infezione, il neighbor viene contagiato
                #quindi gli assegniamo il tempo di recupero e aumentiamo il count dei nodi infettati dal node
                I.add_node(neighbor, state='infected')
                I.add_node(neighbor, recovery_time_left=(t_rec))
                I.add_node(neighbor, color='red')
                I.add_node(node, count_infected=(read_count_infected(node) + 1))
                I.add_node(node, temporary_count_infected=(read_temporary_count_infected(node) + 1))
                
                '''ninf=read_neighbor_infected(node)
                ninf[step].append(neighbor)
                print(ninf)
                I.add_node(node, neighbor_infected=ninf)'''

                new_infected_nodes.append(neighbor)
                
        if read_recovery_time_left(node) >= 1:
            I.add_node(node, recovery_time_left=(read_recovery_time_left(node) - 1))
            if read_recovery_time_left(node) == 0:
                I.add_node(node, state = 'recovered')
                I.add_node(node, immunity_time_left = (t_sus + 1))
                copy_infected_nodes.remove(node)

    statistics = {
        "susceptible": 0,
        "recovered": 0,
        "infected": 0
    }

    turn_spreader = []

    for node in (I.nodes):
        if read_state(node) == 'recovered':
            I.add_node(node, color='green') #coloriamo di verde i nodi guariti
            if read_immunity_time_left(node) >= 1:
                I.add_node(node, immunity_time_left=(read_immunity_time_left(node) - 1))
                if read_immunity_time_left(node) == 0:
                    I.add_node(node, state='susceptible')
                    I.add_node(node, color='blue') #riportiamo al colore iniziale di blu i nodi di nuovo suscettibili    
        
        statistics[read_state(node)] += 1
        #crea tuple con id nodo e valore temporaneo di nodi infettati nel turno, quindi riazzera count
        if read_temporary_count_infected(node) > 0:
            turn_spreader.append((node, read_temporary_count_infected(node)))    
            I.add_node(node, temporary_count_infected=0)

    #sort turn_spreader e si prendono solo i primi 4 nodi di cui fare l'append in lista max_spreader
    turn_spreader.sort(key=lambda a: a[1], reverse=True)

    #Per come è scritta ora vengono appese anche delle liste vuote se sul turno non si contagia nessuno
    if len(turn_spreader) >= 4:
        t_tuple = []

        for i in range(4):
            t_tuple.append(turn_spreader[i])
        
        '''
        count_diff_value = 0
        i = 0
        prev_val = -1
        while (count_diff_value < 4 or i < (len(turn_spreader) - 1)):
            val = turn_spreader[i][1]
            if (val != prev_val):
                count_diff_value += 1
                tmp_val = val
            t_tuple.append(turn_spreader[i])
            i += 1
        '''

        max_spreader.append(t_tuple)
        print(max_spreader)
    else:
        max_spreader.append(turn_spreader)
        print(max_spreader)

    '''for element in I.nodes.data() :
        print(element)'''
    #print("\n\n")
    st_susceptible = statistics['susceptible']
    st_infected = statistics['infected']
    st_recovered = statistics['recovered']

    print_stats(st_susceptible,st_infected, st_recovered)
    st_tuple = (st_susceptible, st_infected, st_recovered)
    statistics_graph.append(st_tuple)
    print("\n\n")

    infected_nodes = copy_infected_nodes + new_infected_nodes

#print(statistics_graph)

#Creiamo una lista per il tempo, lunga quanto le statistiche (quindi uguale a t_step)
time = [i for i in range(len(statistics_graph))]

#Spacchetta tuple e fa plot dei 3 valori

y1, y2, y3 = zip(*statistics_graph)

'''
# Plot senza interpolazione
plt.plot(time, y1, label="S", color='b')
plt.plot(time, y2, label="I", color='r')
plt.plot(time, y3, label="R", color='g')
'''

time = np.array(time)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)

time_y1_spline = make_interp_spline(time, y1)
time_y2_spline = make_interp_spline(time, y2)
time_y3_spline = make_interp_spline(time, y3)
 
# Returns evenly spaced numbers over a specified interval
pl_time = np.linspace(time.min(), time.max(), 500)
pl_y1 = time_y1_spline(pl_time)
pl_y2 = time_y2_spline(pl_time)
pl_y3 = time_y3_spline(pl_time)

# Plotting the Graph
plt.plot(pl_time, pl_y1, label="S", color='b')
plt.plot(pl_time, pl_y2, label="I", color='r')
plt.plot(pl_time, pl_y3, label="R", color='g')

#Aggiungiamo label a ascisse e ordinate; nome al modello e legenda. Quindi mostriamo plot
plt.xlabel('Time Step')
plt.ylabel('Nodes')
plt.title('SIR Model - Disease Trends')
plt.legend()

plt.show()