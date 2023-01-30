import networkx as nx
import pygraphviz as pgv
import pydot
import random
import copy
import csv
import sys
import os

import logging

from datetime import datetime

import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

from operator import itemgetter
from collections import defaultdict


class colors: 
    reset           ='\033[0m'
    bold            ='\033[01m'
    disable         ='\033[02m'

    class text: 
        red         ='\033[31m'
        green       ='\033[32m'
        blue        ='\033[34m'
        yellow      ='\033[93m'


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

    # Attributi grafo di default
    A.graph_attr.update(N.graph.get("graph", {}))
    A.node_attr.update(N.graph.get("node", {}))
    A.edge_attr.update(N.graph.get("edge", {}))

    A.graph_attr.update(
        (k, v) for k, v in N.graph.items() if k not in ("graph", "node", "edge")
    )

    # Aggiunta nodi
    for n, nodedata in N.nodes(data=True):
        A.node_attr['style']='filled'

        A.add_node(n)
        a = A.get_node(n)
        
        for k, v in nodedata.items():
            a.attr.update({k: str(v)})
            if str(k) == 'color':
                a.attr['fillcolor']=str(v)

    # Aggiunta archi
    if N.is_multigraph():
        for u, v, key, edgedata in N.edges(data=True, keys=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items() if k != "key"}
            A.add_edge(u, v, key=str(key))
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)

    else:
        for u, v, edgedata in N.edges(data=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items()}
            A.add_edge(u, v)
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)

    return A



#################################################################
#                                                               #
#                FUNZIONI CONTROLLO CSV DI INPUT                #
#                                                               #
#################################################################

# Controllo se header CSV letto è uguale all'header CSV da noi impostato  -->  ['source','target'] 
def test_header(csv_header):
    if (csv_header[0] != header[0] or csv_header[1] != header[1]):
        errore = colors.text.red + "\nErrore lettura file '" + grafo_csv + "': header non corretto!\n\n" + colors.reset
        sys.exit(errore + errore_lettura_csv)

# Controllo se sulla riga CSV letta ci sono i parametri minimi indispensabili (nodo partenza e destinazione)
def test_len(len_row, num_row, node):
    if ((len_row < 2) or (node.strip() == '')): # 2 = len(header)
        errore = colors.text.red + "\nErrore lettura file '" + grafo_csv + "': alla riga " + str(num_row) + " mancano dei parametri indispensabili!\n\n" + colors.reset
        return errore
    else:
        return ''

# Richiamo funzione test_len() e aggiunta archi al grafo
def test_and_add_edge(row, num_row, warnings):
    raise_sys_exit = False
    errore = ''
    
    try:
        errore = test_len(len(row), num_row, row[1])
        if (errore  != ''):
            raise SystemExit(errore)
        I_reset.add_edge(int(row[0]), int(row[1]))
    except ValueError as ve:
        errore = "Riga " + str(num_row) + " del file '" + grafo_csv + "': l'ID del nodo '" + header[0] + "' o '" + header[1] + "' non è un numero intero!"
        warnings.append(errore)
    except SystemExit as se:
        raise_sys_exit = True
        errore = se
    except Exception as ex:
        print(colors.text.red + "Si è verificato un errore imprevisto durante l'aggiunta degli archi al grafo: " + str(ex) + colors.reset)
    finally:
        if (raise_sys_exit):
            sys.exit(errore)
        return warnings


#################################################################
#                                                               #
#                FUNZIONI CONTROLLO INPUT UTENTE                #
#                                                               #
#################################################################

def test_input_range(value, control, question, min, max):
    if (not control):
        user_value = input(question)
        try:
            user_value = float(user_value)
            if (user_value < min or user_value > max):
                raise ValueError()
            else:
                control = True
                value = user_value
        except ValueError:
            print(colors.bold + colors.text.red + "ERRORE: " + colors.disable + 'devi inserire un valore compreso tra 0 e 1\n' + colors.reset)
        finally:
            return value, control
    else:
        return value, control


def test_input_int(value, control, question):
    if (not control):
        user_value = input(question)
        try:
            user_value = int(user_value)
            control = True
            value = user_value
        except ValueError:
            print(colors.bold + colors.text.red + "ERRORE: " + colors.disable + 'devi inserire un numero intero\n' + colors.reset)
        finally:
            return value, control
    else:
        return value, control


def test_input_scelta(question):
    error = True
    
    while(error):
        value = input(question)
        try:
            value = int(value)
            if (value != 0 and value != 1):
                raise ValueError()
            else:
                error = False
        except ValueError:
            print(colors.bold + colors.text.red + "ERRORE: " + colors.disable + 'puoi scegliere solo 0 o 1\n' + colors.reset)
    return value



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



#################################################################
#                                                               #
#              FUNZIONI PRINCIPALI PER SIMULAZIONE              #
#                                                               #
#################################################################

# Stampa statistiche infezione
def print_stats(sus, inf, rec):
    print('Susceptible: ' + str(sus))
    print('Infected: ' + str(inf))
    print('Recovered: ' + str(rec))
    print("\n")

    logging.info('Susceptible: ' + str(sus))
    logging.info('Infected: ' + str(inf))
    logging.info('Recovered: ' + str(rec))
    logging.info('\n')



# Funzione per infettare percentuale iniziale dei nodi:
def infettainit(grafo, p, e):

    global infected_nodes # Contiene nodi infetti ad ogni turno
    global statistics_graph

    # Reset variabili della simulazione precedente
    infected_nodes = []

    # Numero nodi da infettare
    initinf = int(round(p * (grafo.number_of_nodes()), 0))

    print_stats(grafo.number_of_nodes() - initinf, initinf, 0)

    # Salvataggio tupla per creazione statistiche e grafici SIR
    st_tuple = (grafo.number_of_nodes() - initinf, initinf, 0) 
    statistics_graph[e].append(st_tuple)
    
    while initinf > 0:
        # Scelta casuale di nodi da infettare
        x = random.choice(list(grafo.nodes))

        if read_state(grafo, x) != 'infected':
            # Set attributi nodi
            grafo.add_node(x, state='infected')
            grafo.add_node(x, color='red')
            grafo.add_node(x, recovery_time_left=t_rec)
            
            infected_nodes.append(x)
            non_infected_nodes[e].remove(x)
            
            initinf -= 1

    print('I nodi infettati iniziali sono:\n' + str(infected_nodes) + '\n')
    logging.info('I nodi infettati iniziali sono:\n' + str(infected_nodes) + '\n')



# Funzione per il calcolo di Max Spreader di raggio 2
def calc_infected_neighbors(grafo):
    result = []

    # Per ogni nodo "node" nel grafo
    for node in grafo.nodes :
        time_step = 0
        counter_infected = 0

        # Accedo alla lista "neighbor_infected" (attributo del nodo) che è lunga t_step e contiene 
        # la lista di vicini infettati in un turno 
        for element_t in read_neighbor_infected(grafo, node):
            time_step += 1

            # Scorro tutti i vicini infettati in un turno aggiungendoli al contatore e nell'arco di tempo
            # in cui sono infetti, aggiungo al contatore anche quelli che hanno infettato a loro volta
            for neighbor in element_t:
                counter_infected += 1
                start_time_step = time_step
                stop_time_step = start_time_step + t_rec

                for neighbor_time_step in range(start_time_step, stop_time_step):
                    if stop_time_step > t_step:
                        break
                    counter_infected += len(read_neighbor_infected(grafo, neighbor)[neighbor_time_step])

        # Salvo la tupla (ID_nodo, infettati_raggio_2)
        value = (node, counter_infected)
        result.append(value)

    return result



#################################################################
#                                                               #
#            VARIABILI GLOBALI E SETTING ENVIRONMENT            #
#                                                               #
#################################################################

# Disabilitazione logger matplotlib
logging.getLogger('matplotlib.font_manager').disabled = True

# Variabili di default se l'utente non passa input
grafo_csv = 'graph.csv'
p_init = 0.10
p_trans = 0.15
t_rec = 4
t_sus = 3
t_step = 15
simulations = 4
scelta = 0

# Variabili di controllo input
grafo_csv_OK = p_init_OK = p_trans_OK = t_rec_OK = t_sus_OK = t_step_OK = simulations_OK = False

request_p_init = "Percentuale 'p_init' di nodi iniziali da infettare (valore tra 0-1): "
request_p_trans = "Probabilità 'p_trans' di trasmettere la malattia (valore tra 0-1): "
request_t_sus = "Tempo 't_sus' per passare dallo stato guarito a suscettibile: "
request_t_rec = "Tempo 't_rec' per passare dallo stato infetto a guarito: "
request_t_step = "Durata temporale 't_step' di una simulazione: "
request_simulations = "Numero di simulazioni da effettuare: "

print('\n\nVuoi impostare i seguenti valori di default per la simulazione o inserirli a mano?\n')
print(request_p_init + colors.text.yellow + str(p_init) + colors.reset + '\n' + request_p_trans + colors.text.yellow + str(p_trans) + colors.reset + '\n' + request_t_sus + colors.text.yellow + str(t_sus) + colors.reset + '\n' + request_t_rec + colors.text.yellow + str(t_rec) + colors.reset + '\n' + request_t_step + colors.text.yellow + str(t_step) + colors.reset + '\n' + request_simulations + colors.text.yellow + str(simulations) + colors.reset)

scelta = test_input_scelta('\n> '  + colors.text.yellow + '0' + colors.reset + ' : Default\n> '  + colors.text.yellow + '1' + colors.reset + ' : Imposta manualmente\n\nScelta: ')

if scelta == 1:
    # Controllo input
    while (not (grafo_csv_OK and p_init_OK and p_trans_OK and t_rec_OK and t_sus_OK and t_step_OK and simulations_OK)):
        print('\n\nInserisci i seguenti valori per inizializzare le simulazioni:\n')

        if (not grafo_csv_OK):
            grafo_csv = input("Nome del file csv da leggere: ")
            if (not (os.path.isfile(grafo_csv))):
                print(colors.bold + colors.text.red + "ERRORE: " + colors.disable + 'non esiste nessun file ' + str(grafo_csv) + colors.reset + '\n')
            else:
                grafo_csv_OK = True

        p_init, p_init_OK = test_input_range(p_init, p_init_OK, request_p_init, 0, 1)
        p_trans, p_trans_OK = test_input_range(p_trans, p_trans_OK, request_p_trans, 0, 1)    
        t_sus, t_sus_OK = test_input_int(t_sus, t_sus_OK, request_t_sus)
        t_rec, t_rec_OK = test_input_int(t_rec, t_rec_OK, request_t_rec)
        t_step, t_step_OK = test_input_int(t_step, t_step_OK, request_t_step)
        simulations, simulations_OK = test_input_int(simulations, simulations_OK, request_simulations)


I_reset = nx.Graph() # Conterrà la copia di I da ripristinare dopo ogni termine simulazione
I = '' # Conterrà il grafo da utilizzare a ogni simulazione

CSV_read_warnings = []

data_laboratory = {} # Conterrà le proprietà dei nodi da salvare in CSV
data_laboratory['header'] = ['Node'] 

header = ['source','target']
infected_nodes = []
statistics_graph = []
non_infected_nodes = []
max_spreader_raggio_1 = []
max_spreader_raggio_2 = []

for i in range(simulations):
    statistics_graph.append([])
    non_infected_nodes.append([])
    max_spreader_raggio_1.append([])
    max_spreader_raggio_2.append([])


errore_lettura_csv = 'Il file CSV passato in input deve iniziare con il seguente header:\t\t' + header[0] + ',' + header[1] + ' [, ...]\n'

dir_output_file = 'OutputFile'
path_grafico_attuale = ''

# Creazione cartella che conterrà i file di output
try:
    try:
        os.mkdir(dir_output_file)
    except FileExistsError as error:
        pass
    except OSError as error:
        sys.exit('Si è verificato un errore: ' + str(error))

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_grafico_attuale = dir_output_file + '/' + current_time
    os.mkdir(path_grafico_attuale)

except FileExistsError:
    pass
except Exception as error:
    sys.exit('Si è verificato un errore: ' + str(error))



#################################################################
#                                                               #
#                INIZIO PROGRAMMA - LETTURA CSV                 #
#                                                               #
#################################################################

with open(grafo_csv, encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    # Lettura prima riga (header) CSV
    csv_header = next(csv_reader)
    test_header(csv_header) 

    # Genera un WARNING in caso CSV contenga un grafo diretto
    csv_type = next(csv_reader) 

    if (len(csv_type) > 2 and csv_type[2] == 'directed'):
        CSV_read_warnings.append("Il file '" + grafo_csv + "' contiene un grafo 'diretto' ma verrà trasformato in un grafo 'indiretto'") 

    # Aggiunge l'arco della riga appena letta, sennò andrebbe perso
    CSV_read_warnings = test_and_add_edge(csv_type, 2, CSV_read_warnings)

    num_row = 2
    for row in csv_reader:
        num_row += 1
        CSV_read_warnings = test_and_add_edge(row, num_row, CSV_read_warnings) 

# Rimuove i self loop
I_reset.remove_edges_from(nx.selfloop_edges(I_reset)) 

# Entra nella directory di salvataggio file
os.chdir(path_grafico_attuale)

# Set del logger
logging.basicConfig(filename="log.txt", level=logging.DEBUG, format=None)

if scelta == 0:
    logging.info('Per la simulazione sono stati usati i seguenti valori di default:\n')
else:
    logging.info('Per la simulazione sono stati usati i seguenti valori:\n')
logging.info(request_p_init + str(p_init) + '\n' + request_p_trans + str(p_trans) + '\n' + request_t_sus + str(t_sus) + '\n' + request_t_rec + str(t_rec) + '\n' + request_t_step + str(t_step) + '\n' + request_simulations + str(simulations) + '\n\n')


logging.info("Lettura file '" + grafo_csv + "' in corso...")

# Stampa eventuali errori generati durante lettura CSV
if (len(CSV_read_warnings) > 0):
    print('\n\n' + colors.text.yellow + 'WARNING:')
    logging.warning('\n\nWARNING:')
    for warning in CSV_read_warnings:
        print(warning)
        logging.warning(warning)
    
    print(colors.reset + '\n\n')
    logging.warning('\n\n')
else:
    logging.info("\nLettura file effettuata senza errori!\n\n")


print('\n\n')
#print('Edge list:\n' + str(list(I_reset.edges)) + '\n')
print('Node list:\n' + str(list(I_reset.nodes)) + '\n')
print('Number of nodes:\n' + str(I_reset.number_of_nodes()) + '\n')
print('Number of edges:\n' + str(I_reset.number_of_edges()) + '\n')

#logging.info('Edge list:\n' + str(list(I_reset.edges)) + '\n')
logging.info('Node list:\n' + str(list(I_reset.nodes)) + '\n')
logging.info('Number of nodes:\n' + str(I_reset.number_of_nodes()) + '\n')
logging.info('Number of edges:\n' + str(I_reset.number_of_edges()) + '\n\n')


# Aggiunta attributi nodi
for node in (I_reset.nodes): 
    lst = []
    for i in range(t_step):
        lst.append([])

    I_reset.add_nodes_from([(node, {'state': 'susceptible', 'color': 'blue', 'count_infected': 0, 'temporary_count_infected': 0, 'neighbor_infected': lst, 'recovery_time_left': 0})])

    data_laboratory[node] = [] 
    data_laboratory[node].append(node)

# Conterrà i nodi che non vengono mai infettati
for i in range (simulations):
    non_infected_nodes[i] = list(I_reset)




#################################################################
#                                                               #
#                CORPO PROGRAMMA - STEP CONTAGIO                #
#                                                               #
#################################################################

print('\n\n')
print('######################################################################')
print('#                                                                    #')
print('#                         INIZIO SIMULAZIONI                         #')
print('#                                                                    #')
print('######################################################################')
print('\n')

logging.info('######################################################################')
logging.info('#                                                                    #')
logging.info('#                         INIZIO SIMULAZIONI                         #')
logging.info('#                                                                    #')
logging.info('######################################################################')
logging.info('\n')


for e in range(simulations):

    print('Inizio simulazione: ' + str(e) + '\n')
    logging.info('Inizio simulazione: ' + str(e) + '\n')

    # Riporta il grafo ai valori iniziali
    I = copy.deepcopy(I_reset)

    logging.info('Contagio nodi iniziali...' + '\n')
    infettainit(I, p_init, e)

    # Inizio turni
    for step in range(t_step):
        print('Turno ' + str(e) + '.' + str(step) + ':\n')
        logging.info('\nTurno ' + str(e) + '.' + str(step) + ':\n')

        statistics = {
            "susceptible": 0,
            "recovered": 0,
            "infected": 0
        }

        # Contiene i nodi infettati nel turno
        new_infected_nodes = []
        
        # Facciamo shuffle per dare maggiore casualità alle infezioni, senza avvantaggiare i primi nodi infettati
        random.shuffle(infected_nodes)

        copy_infected_nodes = infected_nodes.copy()

        # Stadio di infezione vicini
        for node in (infected_nodes):
            for neighbor in (I.adj[node]):
                if read_state(I, neighbor) =='susceptible' and round(random.uniform(0.00, 1.00), 2) <= p_trans:
                    I.add_node(neighbor, state='infected')
                    I.add_node(neighbor, recovery_time_left=(t_rec))
                    I.add_node(neighbor, color='red')
                    I.add_node(node, count_infected=(read_count_infected(I, node) + 1))
                    I.add_node(node, temporary_count_infected=(read_temporary_count_infected(I, node) + 1))
                    
                    # Aggiornamento attributo vicini infettati
                    ninf = read_neighbor_infected(I, node)
                    ninf[step].append(neighbor)
                    I.add_node(node, neighbor_infected=ninf)

                    new_infected_nodes.append(neighbor)

                    logging.info('Nodo ' + str(node) + ' ha infettato nodo ' + str(neighbor))

                    try:
                        non_infected_nodes[e].remove(neighbor)
                    except ValueError:
                        pass # Ha provato ad eliminare un nodo già eliminato in precedenza

            # Gestione nodi infettati
            if read_recovery_time_left(I, node) >= 1:
                I.add_node(node, recovery_time_left=(read_recovery_time_left(I, node) - 1))
                if read_recovery_time_left(I, node) == 0:
                    I.add_node(node, state = 'recovered')
                    I.add_node(node, immunity_time_left = (t_sus + 1))
                    copy_infected_nodes.remove(node)

                    logging.info('Nodo ' + str(node) + ' è guarito')

        # Gestione nodi guariti
        for node in (I.nodes):
            if read_state(I, node) == 'recovered':
                I.add_node(node, color='green')
                if read_immunity_time_left(I, node) >= 1:
                    I.add_node(node, immunity_time_left=(read_immunity_time_left(I, node) - 1))
                    if read_immunity_time_left(I, node) == 0:
                        I.add_node(node, state='susceptible')
                        I.add_node(node, color='blue')

                        logging.info('Nodo ' + str(node) + ' è nuovamente suscettibile')
            
            statistics[read_state(I, node)] += 1
            
            # Salva il numero di nodi infettati nel turno per ogni nodo
            if ((read_temporary_count_infected(I, node)) > 0):
                max_spreader_raggio_1[e].append((node, read_temporary_count_infected(I, node)))
                I.add_node(node, temporary_count_infected=0)
        
        logging.info('')

        st_susceptible = statistics['susceptible']
        st_infected = statistics['infected']
        st_recovered = statistics['recovered']
        print_stats(st_susceptible, st_infected, st_recovered)

        # Salvataggio tupla per creazione statistiche e grafici SIR
        st_tuple = (st_susceptible, st_infected, st_recovered)
        statistics_graph[e].append(st_tuple)

        # Aggiornamento degli infetti per il turno successivo
        infected_nodes = copy_infected_nodes + new_infected_nodes


    # Calcolo Max Spreader raggio 2
    max_spreader_raggio_2[e] = calc_infected_neighbors(I) 
    max_spreader_raggio_2[e].sort(key=lambda a: a[1], reverse=True)


    ################################################################
    #                                                              #
    #                 PLOT STATISTICHE SIMULAZIONE                 #
    #                                                              #
    ################################################################

    # Creazione asse tempo per plot
    time = [i for i in range(len(statistics_graph[e]))]

    # Spacchettamento tuple
    y1, y2, y3 = zip(*statistics_graph[e])

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
    plt.ylabel('# Nodes')
    plt.title('SIR Model - Disease Trends - Simulation ' + str(e))
    plt.legend()



    ###############################################################
    #                                                             #
    #                  CREAZIONE RENDERING GRAFO                  #
    #                                                             #
    ###############################################################

    # Conversione grafo networkx in pygraphviz
    A = my_version_to_agraph(I)
    A.layout(prog='sfdp')
    A.draw('grafico_finale' + str(e) + '.svg')

    plt.savefig('stat_plot' + str(e) + '.png')
    plt.clf()
    
    if (e != (simulations - 1)) : print("\n------------------------------------------------------------------------------------\n\n")
    if (e != (simulations - 1)) : logging.info("\n------------------------------------------------------------------------------------\n\n")


plt.close() # Dopo aver salvato vari file conviene rilasciare la memoria, anche per evitare bug

print('\n')
print('######################################################################')
print('#                                                                    #')
print('#             FINE SIMULAZIONI - INIZIO CALCOLO SPREADER             #')
print('#                                                                    #')
print('######################################################################')
print('\n')

logging.info('')
logging.info('######################################################################')
logging.info('#                                                                    #')
logging.info('#             FINE SIMULAZIONI - INIZIO CALCOLO SPREADER             #')
logging.info('#                                                                    #')
logging.info('######################################################################')
logging.info('\n')


logging.info('Calcolo nodo massimo spreader (R1 e R2) per singola simulazione\n')

percentages_list = []
counter_simulazioni = -1

# Calcolo misurazioni per ogni simulazione
for simulation_value in max_spreader_raggio_1:
    counter_simulazioni += 1
    
    # Crea dizionario che conterrà il totale dei nodi infettati da un nodo nella simulazione
    total_infected_from_node = defaultdict(int)
    
    # Somma tutti i valori con stessa key dentro al dizionario
    total_infections = 0
    for i, k in simulation_value:
        total_infected_from_node[i] += k
        total_infections += k

    total_infected_from_node_list = list(total_infected_from_node.items())
    total_infected_from_node_list.sort(key=lambda a: a[1], reverse=True)

    # Aggiunta nodi che non hanno mai contagiato nessuno
    missing_nodes = list(I_reset)
    for tupla in total_infected_from_node_list:
        missing_nodes.remove(tupla[0])    
    for node in missing_nodes:
        total_infected_from_node_list.append((node, 0))


    print('Simulazione: ' + str(counter_simulazioni) + '\n\n')
    print('Numero contagi: ' + str(total_infections) + '\n')
    print('Max Spreader raggio 1:\n' + str(total_infected_from_node_list) + '\n')
    print('Max Spreader raggio 2:\n' + str(max_spreader_raggio_2[counter_simulazioni]) + '\n')
    print('Nodi mai infettati:\n' + str(non_infected_nodes[counter_simulazioni]) + '\n')
    print('\n-------------------------------------------------------------------------------------\n\n')

    logging.info('Simulazione: ' + str(counter_simulazioni) + '\n\n')
    logging.info('Numero contagi: ' + str(total_infections) + '\n')
    logging.info('Max Spreader raggio 1:\n' + str(total_infected_from_node_list) + '\n')
    logging.info('Max Spreader raggio 2:\n' + str(max_spreader_raggio_2[counter_simulazioni]) + '\n')
    logging.info('Nodi mai infettati:\n' + str(non_infected_nodes[counter_simulazioni]) + '\n')
    logging.info('\n-------------------------------------------------------------------------------------\n\n')

    # Calcolo valore percentuale di nodi infettati da uno specifico nodo rispetto 
    # il totale di nodi infettati all'interno di una simulazione
    for tupla in total_infected_from_node_list:
        tupla_list = list(tupla)
        tupla_list.append(round(100 * (tupla[1]/total_infections), 2))
        percentages_list.append(tuple(tupla_list))



print('Calcolo nodo massimo spreader (R1 e R2) tra tutte le simulazioni (basato su media)\n')
logging.info('Calcolo nodo massimo spreader (R1 e R2) tra tutte le simulazioni (basato su media)\n')



# CALCOLO MAX SPREADER R1

# Crea dizionario che conterrà media delle percentuali di infezione R1 tra le simulazioni
avg_percentage = defaultdict(int)

# Somma i valori delle percentuali con stessa key dentro al dizionario
for i, k, p in percentages_list:
    avg_percentage[i] += p

# Prepariamo header e dati per il salvataggio su un futuro file CSV
data_laboratory['header'].append('R1 AVG % Infected Nodes')
for i in avg_percentage:
    avg_percentage[i] = round(avg_percentage[i]/simulations, 2) # Calcola media tra i valori percentuali delle varie simulazioni
    data_laboratory[i].append(avg_percentage[i])

avg_percentage_list = list(avg_percentage.items())
avg_percentage_list.sort(key=lambda a: a[1], reverse=True)

print('R1 AVG % Infected Nodes:\n' + str(avg_percentage_list) + '\n\n')
logging.info('R1 AVG % Infected Nodes:\n' + str(avg_percentage_list) + '\n\n')





# CALCOLO MAX SPREADER R2

temp = []
for simulation_value in max_spreader_raggio_2:
    temp += simulation_value

# Crea dizionario che conterrà i valori di infezioni R2 tra le simulazioni
total_sum_R2 = defaultdict(int)

# Somma i valori delle infezioni con stessa key dentro al dizionario
for i, k in temp:
    total_sum_R2[i] += k

# Prepariamo header e dati per il salvataggio su un futuro file CSV
data_laboratory['header'].append('R2 AVG Infected Nodes')
for i in total_sum_R2:
    total_sum_R2[i] = round(total_sum_R2[i]/simulations, 2) # Calcola media tra i valori delle varie simulazioni
    data_laboratory[i].append(total_sum_R2[i])

total_sum_R2_list = list(total_sum_R2.items())
total_sum_R2_list.sort(key=lambda a: a[1], reverse=True)

print('R2 AVG Infected Nodes:\n' + str(total_sum_R2_list) + '\n\n')  # MEDIA DI NODI DI SECONDO GRADO INFETTATI TRA LE VARIE SIMULAZIONI
logging.info('R2 AVG Infected Nodes:\n' + str(total_sum_R2_list) + '\n\n')




# Calcolo nodi mai infettati tra tutte le simulazioni 
never_infected_nodes = []
if simulations > 1:
    for i in range (simulations):
        if i == 0:
            never_infected_nodes = list(set(non_infected_nodes[i]).intersection(non_infected_nodes[i+1]))
        elif i < (simulations - 1):
            never_infected_nodes = list(set(never_infected_nodes).intersection(non_infected_nodes[i+1]))
else:
    never_infected_nodes = non_infected_nodes[0]

if (len(never_infected_nodes) == 0):
    print("Nessun nodo ha evitato l'infezione\n\n")
    logging.info("Nessun nodo ha evitato l'infezione\n\n")
else:
    print('I nodi che non sono mai stati infettati tra tutte le simulazioni sono:\n' + str(never_infected_nodes) + '\n\n')
    logging.info('I nodi che non sono mai stati infettati tra tutte le simulazioni sono:\n' + str(never_infected_nodes) + '\n\n')

data_laboratory['header'].append('Never infected')
for key in data_laboratory:
    if key == 'header':
        continue

    if key in never_infected_nodes:
        data_laboratory[key].append(1)
    else:
        data_laboratory[key].append(0)

logging.info('Data laboratory:\n' + str(data_laboratory) + '\n\n')


# Creazione file CSV contenenti i nodi e i relativi valori delle misurazioni effettuate
with open('data_laboratory.csv', 'w') as f:
    write = csv.writer(f)
    
    for key in data_laboratory:
        write.writerow(data_laboratory[key])


###############################################################
#                                                             #
#           CREAZIONE PLOT MISURAZIONI MAX SPREADER           #
#                                                             #
###############################################################

fig = plt.figure()

# Plot Max Spreader raggio 1
plt.subplot(2, 1, 1)
plt.xlabel('Nodes')
plt.ylabel('AVG % Infected Nodes')
plt.title('Radius 1 Infection Measure - Over all simulations')

x, y = zip(*avg_percentage_list)

X = np.array(x)
Y = np.array(y)

plt.scatter(X, Y, 5)
plt.grid()


# Plot Max Spreader raggio 2
plt.subplot(2, 1, 2)
plt.xlabel('Nodes')
plt.ylabel('AVG Infected Nodes')
plt.title('Radius 2 Infection Measure - Over All Simulations')

x, y = zip(*total_sum_R2_list)

X = np.array(x)
Y = np.array(y)

plt.scatter(X, Y, 5)
plt.grid()


fig.tight_layout() # Aggiusta i margini tra i vari sublots
fig.savefig('MaxSpreaderMeasure.png')

print("Nella cartella '" + path_grafico_attuale + "' sono stati salvati i seguenti file:")
ordered_file = sorted(os.listdir())
for file in ordered_file:
    print('  - ' + str(file))

plt.show()
plt.clf()