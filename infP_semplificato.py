import networkx as nx
import random
import csv
import sys


#################################################################
#                                                               #
#            VARIABILI GLOBALI E SETTING ENVIRONMENT            #
#                                                               #
#################################################################

I = nx.Graph()

header = ['source','target']
infected_nodes = []

errore_lettura_csv = 'Il file CSV passato in input deve iniziare con il seguente header:\t\t' + header[0] + ',' + header[1] + ' [, ...]\n'

#grafo_csv = input("Inserisci il nome del file csv da leggere: ")

#valori iniziali
grafo_csv = 'graph2.csv'
p_init = 0.20
p_trans = 0.10
t_rec = 7
t_sus = 15
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
    print('1 -- I.number_of_nodes(): ' + str(I.number_of_nodes())) 
    print(I)
    print('Edge list:\n' + str(list(I.edges)) + '\n')
    print('Node list:\n' + str(list(I.nodes)) + '\n')
           
    initinf = round(p * (I.number_of_nodes()), 0)
    print('initinf: ' + str(initinf))
    while initinf > 0:
        print('2 -- I.number_of_nodes(): ' + str(I.number_of_nodes()))
        x = random.randint(0, I.number_of_nodes() - 1)
        print('x: ' + str(x) + '\n\n')

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
        test_len(len(row), num_row)

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

for x in range(I.number_of_nodes()):
    I.add_nodes_from([(x, {'state': 'susceptible', 'color': 'blue', 'count_infected': 0})])

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
                new_infected_nodes.append(neighbor)
                
        if read_recovery_time_left(node) >= 1:
            I.add_node(node, recovery_time_left=(read_recovery_time_left(node) - 1))
            if read_recovery_time_left(node) == 0:
                I.add_node(node, state = 'recovered')
                I.add_node(node, immunity_time_left = (t_sus + 1))
                copy_infected_nodes.remove(node)

    for node in (I.nodes):
        if read_state(node) == 'recovered':
            I.add_node(node, color='green') #coloriamo di verde i nodi guariti
            if read_immunity_time_left(node) >= 1:
                I.add_node(node, immunity_time_left=(read_immunity_time_left(node) - 1))
                if read_immunity_time_left(node) == 0:
                    I.add_node(node, state='susceptible')
                    I.add_node(node, color='blue') #riportiamo al colore iniziale di blu i nodi di nuovo suscettibili    
    
    for element in I.nodes.data() :
        print(element)
    print("\n\n")

    infected_nodes = copy_infected_nodes + new_infected_nodes

