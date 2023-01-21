import networkx as nx
import pygraphviz as pgv
import csv
import sys


from timeit import default_timer #timer prova
import pydot


#grafo_csv = input("Inserisci il nome del file csv da leggere: ")
grafo_csv = 'graph3.csv'

G = ''
G_type = ''

errore_lettura_csv = 'Il file csv passato in input deve avere il seguente header:\t\tsource,target,type\n'
errore_lettura_csv += 'Nella colonna type sono accettati solo i seguenti valori:\t\t"directed" o "undirected"'


def test_type(edge_type, num_row):
    if (edge_type != 'directed' and edge_type != 'undirected'):
        errore = 'Errore lettura file ' + grafo_csv + ': valore colonna "type" = "' + edge_type + '", riga ' + str(num_row) + ', non corretto!\n\n'
        sys.exit(errore + errore_lettura_csv)

    if (edge_type != G_type):
        errore = 'Errore lettura file ' + grafo_csv + ': valore colonna "type", riga ' + str(num_row) + ', = "' + edge_type + '" ma tipologia grafo in uso = "' + G_type + '"!\n\n'
        sys.exit(errore + errore_lettura_csv)

def test_len(len_row, num_row):
    if (len_row < 3):
        errore = 'Errore lettura file ' + grafo_csv + ': alla riga ' + str(num_row) + ' mancano dei parametri indispensabili!\n\n'
        sys.exit(errore)


with open(grafo_csv, encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    csv_header = next(csv_reader)
    print(str(csv_header) + '\n\n')

    if (csv_header[0] != 'source' or csv_header[1] != 'target' or csv_header[2] != 'type'):
        errore = 'Errore lettura file ' + grafo_csv + ': header non corretto!\n\n'
        sys.exit(errore + errore_lettura_csv)

    csv_type = next(csv_reader)
    print(csv_type[2] + '\n\n')
    print(str(csv_type))

    if (csv_type[2] == 'directed'):
        G = nx.DiGraph()
        G_type = 'directed'
        G.add_edge(csv_type[0], csv_type[1])

    elif (csv_type[2] == 'undirected'):
        G = nx.Graph()
        G_type = 'undirected'
        G.add_edge(csv_type[0], csv_type[1])

    else:
        errore = 'Errore lettura file ' + grafo_csv + ': valore colonna "type" = "' + csv_type[2] + '", riga 2, non corretto!\n\n'
        sys.exit(errore + errore_lettura_csv)


    num_row = 2
    for row in csv_reader:
        num_row += 1
        print(row)
        test_type(row[2], num_row)
        test_len(len(row), num_row)

        G.add_edge(row[0], row[1])


print('\n\n')
print('Edge list:\n' + str(list(G.edges)) + '\n')
print('Node list:\n' + str(list(G.nodes)) + '\n')



#################################################################################################


start_1 = default_timer()
A = nx.nx_agraph.to_agraph(G)
trascorso_1 = default_timer() - start_1
print('Azione 1 ha impiegato: ' + str(trascorso_1) + ' secondi')

start_2 = default_timer()
A.layout(prog='sfdp')
##nx.drawing.nx_pydot.write_dot(G, 'C:\\Users\\Cristian')
###A = nx.drawing.nx_pydot.to_pydot(G)
trascorso_2 = default_timer() - start_2
print('Azione 2 ha impiegato: ' + str(trascorso_2) + ' secondi')

start_3 = default_timer()
A.draw('test_facebook_csv.svg')
###A.write_png('output.png')
trascorso_3 = default_timer() - start_3
print('Azione 3 ha impiegato: ' + str(trascorso_3) + ' secondi')