# SNA-Project

## **Indice**

- [Studenti](#studenti)
- [Informazioni progetto](#info_progetto)
	- [Testo](#testo)
	- [Richieste](#richieste)


# **Studenti**
- Castaldelli Luca
- Cerami Cristian

# **Informazioni progetto**
## Titolo
Epidemic Super Spreaders in Networks - Progetto 1
## Area
Social Graph and Complex Network
## Testo
Lo scopo è quello di implementare un sistema per simulare la diffusione epidemica in una rete complessa e per rilevare i nodi super spreader.<br>
Nel modello i nodi rappresentano individui che, per ogni passo temporale, possono essere infettati dal contatto con i loro nodi vicini nella rete.<br>
Per la diffusione dell'infezione viene adottata una variante del modello SIR (Susceptible Infected Recover), dove ad ogni passo temporale viene calcolata la diffusione epidemica rispetto ai parametri di configurazione del sistema.<br>
In particolare si ha che:
- Tutti i nodi sono inizialmente in stato `suscettibile (susceptible)` e possono diventare infetti
- Una percentuale `p_init` di nodi iniziali viene messa nello stato `infetto (infected)`
- Se un nodo diventa infetto, passa allo stato `guarito (recovered)` dopo un arco temporale di `t_rec` e nel mentre non può essere infettato
- Un nodo guarito diventa nuovamente suscettibile dopo un arco temporale `t_sus`
- Ad ogni step temporale `t_step`, per ogni nodo infetto, vi è una probabilità `p_trans` di trasmettere la malattia a ciascuno dei suoi vicini non infetti
- I cambi di stato dei nodi sono `sincroni` ed avvengono tutti alla fine dello step temporale

## Richieste
Il sistema dovrebbe consentire:
- Il caricamento di qualsiasi grafo in formato CSV
- La configurazione dei parametri
- L'avvio ed esecuzione di una o più simulazioni con gli stessi stati iniziali del nodo
- La visualizzazione delle statistiche e metriche
- La visualizzazione e una rappresentazione grafica del grafo prima, durante o alla fine della simulazione colorando i nodi in base al loro stato
- Il candidato deve sviluppare una tecnica che, attraverso una serie di simulazioni casuali, permette di trovare e mostrare i nodi `super spreader` più rilevanti del grafo passato in input

