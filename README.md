# SNA-Project

## **Indice**

- [Studenti](#studenti)
- [Informazioni progetto](#info_progetto)
	- [Testo](#testo)
	- [Richieste](#richieste)
- [Installazione](#installazione)


# **Studenti**
- Castaldelli Luca
- Cerami Cristian


# **Informazioni progetto**

## Titolo
Epidemic Super Spreaders in Networks - Progetto 1

## Area
Social Graph and Complex Network

## Testo
Lo scopo è quello di implementare un sistema per simulare la diffusione di un'epidemia in una rete complessa e di rilevarne i nodi Super Spreader.<br>
Nel modello i nodi rappresentano individui che, per ogni passo temporale, possono essere infettati dal contatto con i loro nodi vicini nella rete.<br>
Per la diffusione dell'infezione viene adottata una variante del modello SIR (Susceptible Infected Recover), dove ad ogni passo temporale viene calcolata la diffusione epidemica rispetto ai parametri di configurazione del sistema.<br>
In particolare si ha che:
- Tutti i nodi sono inizialmente in stato `suscettibile (susceptible)` e possono diventare infetti
- Una percentuale `p_init` di nodi iniziali viene messa in stato `infetto (infected)`
- Se un nodo diventa infetto, passa allo stato `guarito (recovered)` dopo un arco temporale di `t_rec` e nel mentre non può essere infettato
- Un nodo guarito diventa nuovamente suscettibile dopo un arco temporale `t_sus`
- Ad ogni step temporale `t_step`, per ogni nodo infetto, vi è una probabilità `p_trans` di trasmettere la malattia a ciascuno dei suoi vicini non infetti
- I cambi di stato dei nodi sono `sincroni` ed avvengono tutti alla fine dello step temporale

## Richieste
Il sistema dovrebbe consentire:
- Il caricamento di qualsiasi grafo in formato CSV, sottoforma di lista di archi
- La configurazione dei parametri
- L'avvio ed esecuzione di una o più simulazioni con gli stessi stati iniziali del nodo
- La visualizzazione delle statistiche e metriche
- La visualizzazione e una rappresentazione grafica del grafo prima, durante o alla fine della simulazione colorando i nodi in base al loro stato
- Il candidato deve sviluppare una tecnica che, attraverso una serie di simulazioni casuali, permetta di trovare e mostrare i nodi `super spreader` più rilevanti del grafo passato in input


# **Installazione**
Questa installazione è per <b>Linux</b>.<br>
Per <b>Windows</b> attualmente vi è un problema con il package [PyGraphviz](https://pygraphviz.github.io/documentation/stable/install.html) che a sua volta si basa sul software [Graphviz](https://graphviz.org/download/) che a sua volta basa il funzionamento di alcuni algoritmi implementati (come quello per la rimozione dell'overlap) su alcune librerie che purtroppo sono compatibili solo con Linux. Questo problema, rilevato per la prima volta nel 2017, risulta essere ancora un `open issue` nel relativo [repository Github](https://github.com/ellson/MOTHBALLED-graphviz/issues/1269).

## Passaggi
Prima di scaricare questo repository è consigliato creare un ambiente virtuale sul quale verranno installati i packages necessari.<br>
Se vuoi creare un ambiente virtuale esegui tutti gli step, altrimenti salta al punto 6.<br>
Apri un terminale ed esegui:
1) `python3 -m venv /path/to/<nome-virtual-envirnment>`
2) `cd /path/to/<nome-virtual-envirnment>`
3) `source bin/activate`
4) `git clone https://github.com/Meht-evaS/SNA-Project.git`
5) `cd SNA-Project`
6) `chmod +x install.sh`
7) `./install.sh`