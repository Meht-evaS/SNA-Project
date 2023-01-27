import networkx as nx  # pip3 install networkx
import pygraphviz as pgv  # sudo apt-get install graphviz graphviz-dev
import pydot  # pip3 install pydot
import random
import copy
import csv
import sys
import os

from datetime import datetime

# Rendering curve tramite interpolazione 
import numpy as np  # pip3 install numpy
from scipy.interpolate import make_interp_spline # pip3 install scipy
import matplotlib.pyplot as plt  # pip3 install matplotlib
                                 # Se in esecuzione codice genera errori: sudo apt-get install python3-tk

from math import log
from operator import itemgetter



#################################################################
#                                                               #
# OVERRIDE FUNZIONE DI CONVERSIONE GRAFI NETWORKX IN PYGRAPHVIZ #
#                                                               #
#################################################################

'''
Ne facciamo l'override in quanto abbiamo dovuto aggiungere i seguenti attributi:
    - Attributi del grafo:
        - overlap='false' :
            - Permette di non avere l'overlap dei nodi
        - node_attr['style']='filled' :
            - Permette di riempire il corpo dei nodi con un colore
    - Attributi dei nodi:
        - attr['fillcolor']=colore :
            - Permette di specificare il colore con cui riempire il corpo del nodo
'''

def my_version_to_agraph(N):
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(
            "requires pygraphviz " "http://pygraphviz.github.io/"
        ) from err
    directed = N.is_directed()
    strict = nx.number_of_selfloops(N) == 0 and not N.is_multigraph()
    A = pygraphviz.AGraph(name=N.name, strict=strict, directed=directed, overlap='false')

    # default graph attributes
    A.graph_attr.update(N.graph.get("graph", {}))
    A.node_attr.update(N.graph.get("node", {}))
    A.edge_attr.update(N.graph.get("edge", {}))

    A.graph_attr.update(
        (k, v) for k, v in N.graph.items() if k not in ("graph", "node", "edge")
    )

    # add nodes
    for n, nodedata in N.nodes(data=True):
        A.node_attr['style']='filled'
        #A.node_attr['fillcolor']="black"  #Da un unico colore a tutti i nodi

        A.add_node(n)
        # Add node data
        a = A.get_node(n)
        #a.attr.update({k: str(v) for k, v in nodedata.items()})
        
        #print('nodedata.items() : ' + str(nodedata.items()))
        for k, v in nodedata.items():
            a.attr.update({k: str(v)})
            #print(str(k) + ' : ' + str(v))
            if str(k) == 'color':
                a.attr['fillcolor']=str(v)

    # loop over edges
    if N.is_multigraph():
        for u, v, key, edgedata in N.edges(data=True, keys=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items() if k != "key"}
            A.add_edge(u, v, key=str(key))
            # Add edge data
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)

    else:
        for u, v, edgedata in N.edges(data=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items()}
            A.add_edge(u, v)
            # Add edge data
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)

    return A




#################################################################
#                                                               #
#            VARIABILI GLOBALI E SETTING ENVIRONMENT            #
#                                                               #
#################################################################

#valori iniziali
#grafo_csv = 'Prove_mod_cris\graph.csv'
grafo_csv = 'graph3.csv'
p_init = 0.10
p_trans = 0.15
t_rec = 3
t_sus = 7
t_step = 5
simulations = 4


I_reset = nx.Graph() # Conterrà la copia di I da ripristinare dopo ogni termine simulazione
I = '' # Conterrà il grafo da utilizzare a ogni simulazione

header = ['source','target']
infected_nodes = []
statistics_graph = []
max_spreader_primo_grado = []
max_spreader_secondo_grado = []

for i in range(simulations):
    statistics_graph.append([])
    max_spreader_primo_grado.append([])
    max_spreader_secondo_grado.append([])


errore_lettura_csv = 'Il file CSV passato in input deve iniziare con il seguente header:\t\t' + header[0] + ',' + header[1] + ' [, ...]\n'

#grafo_csv = input("Inserisci il nome del file csv da leggere: ")

dir_output_grafici = 'Grafici'
path_grafico_attuale = ''

try:
    try:
        os.mkdir(dir_output_grafici)
    except FileExistsError as error:
        pass
    except OSError as error:
        sys.exit('Si è verificato un errore: ' + str(error))

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_grafico_attuale = dir_output_grafici + '/' + current_time
    os.mkdir(path_grafico_attuale)

except FileExistsError:
    pass
except Exception as error:
    sys.exit('Si è verificato un errore: ' + str(error))



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
        I_reset.add_edge(int(row[0]), int(row[1]))
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

def read_state(grafo, node):
    return nx.get_node_attributes(grafo, "state")[node]

def read_recovery_time_left(grafo, node):
    return nx.get_node_attributes(grafo, "recovery_time_left")[node]

def read_immunity_time_left(grafo, node):
    return nx.get_node_attributes(grafo, "immunity_time_left")[node]

def read_count_infected(grafo, node):
    return nx.get_node_attributes(grafo, "count_infected")[node]

def read_temporary_count_infected(grafo, node):
    return nx.get_node_attributes(grafo, "temporary_count_infected")[node]

def read_neighbor_infected(grafo, node):
    return nx.get_node_attributes(grafo, "neighbor_infected")[node]

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
def infettainit(grafo, p, e):

    global infected_nodes
    global statistics_graph

    infected_nodes = [] # Reset variabili della simulazione precedente 


    '''print('1 -- grafo.number_of_nodes(): ' + str(grafo.number_of_nodes())) 
    print(grafo)
    print('Edge list:\n' + str(list(grafo.edges)) + '\n')
    print('Node list:\n' + str(list(grafo.nodes)) + '\n')'''
           
    initinf = int(round(p * (grafo.number_of_nodes()), 0))
    #print('initinf = ' + str(initinf) + '  ,  int(initinf) = ' + str(int(initinf)) + '\n')

    print('Susceptible: ' + str(grafo.number_of_nodes() - initinf))
    print('Infected: ' + str(initinf))
    print('Recovered: 0')
    st_tuple = (grafo.number_of_nodes() - initinf, initinf, 0)
    statistics_graph[e].append(st_tuple)
    while initinf > 0:
        #print('2 -- grafo.number_of_nodes(): ' + str(grafo.number_of_nodes()))
        x = random.choice(list(grafo.nodes))
        #print('x: ' + str(x) + '\n\n')

        if read_state(grafo, x) != 'infected':
            grafo.add_node(x, state='infected')
            grafo.add_node(x, color='red')
            grafo.add_node(x, recovery_time_left=t_rec)
            initinf -= 1
            infected_nodes.append(x) 



#################################################################
#                                                               #
#          FUNZIONE CALCOLO INFETTATI DI SECONDO GRADO          #
#                                                               #
#################################################################

def calc_infected_neighbors(grafo):
    result = []
    for node in grafo.nodes :
        time_step = 0
        counter_infected = 0

        for element_t in read_neighbor_infected(grafo, node):
            time_step += 1

            for neighbor in element_t:
                counter_infected += 1
                start_time_step = time_step
                stop_time_step = start_time_step + t_rec

                for neighbor_time_step in range(start_time_step, stop_time_step):
                    if stop_time_step > t_step:
                        break
                    #print('neighbor:' + str(neighbor) + '   --   neighbor_time_step: ' + str(neighbor_time_step))
                    #print('read_neighbor_infected(grafo, neighbor): ' + str(read_neighbor_infected(grafo, neighbor)))
                    counter_infected += len(read_neighbor_infected(grafo, neighbor)[neighbor_time_step])

        value = (node, counter_infected)
        result.append(value)

    return result

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


    I_reset.remove_edges_from(nx.selfloop_edges(I_reset)) # rimuovo i self loop

    if (len(warnings) > 0):
        print('\n\nWARNING:')
        for warning in warnings:
            print(warning)
        
        print('\n\n')


print('\n\n')
print('Edge list:\n' + str(list(I_reset.edges)) + '\n')
print('Node list:\n' + str(list(I_reset.nodes)) + '\n')
print('Number of nodes:\n' + str(I_reset.number_of_nodes()) + '\n')


# I_reset = nx.fast_gnp_random_graph(15, 20, seed=None, directed=False)

for node in (I_reset.nodes): 
    lst = []
    for i in range(t_step):
        lst.append([])

    I_reset.add_nodes_from([(node, {'state': 'susceptible', 'color': 'blue', 'count_infected': 0, 'temporary_count_infected': 0, 'neighbor_infected': lst, 'recovery_time_left': 0})])


 


#################################################################
#                                                               #
#                CORPO PROGRAMMA - STEP CONTAGIO                #
#                                                               #
#################################################################

for e in range(simulations):

    I = copy.deepcopy(I_reset)

    infettainit(I, p_init, e)

    '''for element in I.nodes.data() :
                    print(element)'''

    print("\n\n------------------------------------------------------------------------------------\n\n")


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
                if read_state(I, neighbor) =='susceptible' and round(random.uniform(0.00, 1.00), 2) <= p_trans:
                    #se un node è infettato, prendiamo tutti i vicini infettabili (suscettibili)
                    #prendiamo la probabilità di infezione e un numero random tra 0 e 1
                    #se il numero è minore della probabilità di infezione, il neighbor viene contagiato
                    #quindi gli assegniamo il tempo di recupero e aumentiamo il count dei nodi infettati dal node
                    I.add_node(neighbor, state='infected')
                    I.add_node(neighbor, recovery_time_left=(t_rec))
                    I.add_node(neighbor, color='red')
                    I.add_node(node, count_infected=(read_count_infected(I, node) + 1))
                    I.add_node(node, temporary_count_infected=(read_temporary_count_infected(I, node) + 1))
                    
                    ninf = read_neighbor_infected(I, node)
                    ninf[step].append(neighbor)
                    print('Nodo ' + str(node) + ' ha infettato nodo ' + str(neighbor))
                    #print(ninf)
                    I.add_node(node, neighbor_infected=ninf)

                    new_infected_nodes.append(neighbor)
                    
            if read_recovery_time_left(I, node) >= 1:
                I.add_node(node, recovery_time_left=(read_recovery_time_left(I, node) - 1))
                if read_recovery_time_left(I, node) == 0:
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
            if read_state(I, node) == 'recovered':
                I.add_node(node, color='green') #coloriamo di verde i nodi guariti
                if read_immunity_time_left(I, node) >= 1:
                    I.add_node(node, immunity_time_left=(read_immunity_time_left(I, node) - 1))
                    if read_immunity_time_left(I, node) == 0:
                        I.add_node(node, state='susceptible')
                        I.add_node(node, color='blue') #riportiamo al colore iniziale di blu i nodi di nuovo suscettibili    
            
            statistics[read_state(I, node)] += 1
            #crea tuple con id nodo e valore temporaneo di nodi infettati nel turno, quindi riazzera count
            if read_temporary_count_infected(I, node) > 0:
                turn_spreader.append((node, read_temporary_count_infected(I, node)))    
                I.add_node(node, temporary_count_infected=0)

        #sort turn_spreader e si prendono solo i primi 4 nodi di cui fare l'append in lista max_spreader_primo_grado
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

            max_spreader_primo_grado[e].append(t_tuple)
            #print(max_spreader_primo_grado[e])
        else:
            max_spreader_primo_grado[e].append(turn_spreader)
            #print(max_spreader_primo_grado[e])

        '''for element in I.nodes.data() :
            print(element)'''
        #print("\n\n")
        st_susceptible = statistics['susceptible']
        st_infected = statistics['infected']
        st_recovered = statistics['recovered']

        print_stats(st_susceptible,st_infected, st_recovered)
        st_tuple = (st_susceptible, st_infected, st_recovered)
        statistics_graph[e].append(st_tuple)
        print("\n\n")

        infected_nodes = copy_infected_nodes + new_infected_nodes

    #print(statistics_graph[e])

    #Creiamo una lista per il tempo, lunga quanto le statistiche (quindi uguale a t_step)
    time = [i for i in range(len(statistics_graph[e]))]

    #Spacchetta tuple e fa plot dei 3 valori

    y1, y2, y3 = zip(*statistics_graph[e])

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
    plt.title('SIR Model - Disease Trends - Simulation ' + str(e))
    plt.legend()


    max_spreader_secondo_grado[e] = calc_infected_neighbors(I) 
    #print('\n\n' + str(max_spreader_secondo_grado[e]))
    max_spreader_secondo_grado[e].sort(key=lambda a: a[1], reverse=True)
    print('\n\n' + str(max_spreader_secondo_grado[e]))

    if e == 0: # Va cambiata dir solo al primo turno, prima di iniziare a salvare i vari grafici
        os.chdir(path_grafico_attuale)

    A = my_version_to_agraph(I)
    A.layout(prog='sfdp')
    A.draw('grafico_finale' + str(e) + '.svg')

    plt.savefig('stat_plot' + str(e) + '.png')
    plt.clf()



# Trova un modo per calcolare max spreader usando i valori ottenuti dalle varie simulazioni
print('\n\n\n')
print('statistics_graph :\n' + str(statistics_graph) + '\n\n')
print('max_spreader_primo_grado :\n' + str(max_spreader_primo_grado) + '\n\n')
print('max_spreader_secondo_grado :\n' + str(max_spreader_secondo_grado) + '\n\n')
