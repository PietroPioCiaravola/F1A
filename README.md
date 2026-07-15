# Progetto Guida Autonoma F1 - Tracciato di Monza (CARLA Simulator)

Progetto per l'esame di **Fondamenti di Intelligenza Artificiale** (3° Anno - 1° Semestre). Il codice implementa la connessione al simulatore CARLA e la gestione del tracciato di Monza utilizzando una struttura a pacchetti modulare.

# 🧠 Modello RL: Multi-Agent Proximal Policy Optimization (MAPPO)

Il sistema adotta un approccio Centralized Training with Decentralized Execution (CTDE), ideale per la gestione di scenari competitivi e cooperativi di guida autonoma:

### 1. Actor (Decentralizzato - `src/models.py`)

Ogni veicolo agisce come un agente autonomo guidato da una rete neurale locale:

- **Input (Stato Locale - $18$ dimensioni)**:
  - 16 letture raycast normalizzate per il rilevamento dinamico degli ostacoli e dei limiti fisici del tracciato (con una portata differenziata da 15m a 65m).
  - Velocità lineare dell'auto normalizzata rispetto al limite di riferimento di $v_{max} = 40.0\text{ m/s}$.
  - Angolo di deviazione relativo (normalizzato nell'intervallo $[-1.0, 1.0]$) rispetto alla direzione tangente del waypoint della pista.
- **Output (Spazio delle Azioni - $2$ dimensioni):**
  - Controllo continuo dello sterzo (intervallo $[-1.0, 1.0]$).
  - Controllo continuo della trazione (acceleratore/freno integrato nell'intervallo $[-1.0, 1.0]$).

### 2. Critic (Centralizzato - `src/models.py`)

Utilizzato esclusivamente durante la fase off-line di apprendimento per mitigare i problemi di non-stazionarietà dell'ambiente multi-agente:

- **Input (Stato Globale - $36$ dimensioni):** Concatena gli stati locali di tutti gli agenti attivi nella simulazione (18 input $\times$ 2 veicoli).
- **Output (Valore dello Stato):** Stima del valore dello stato globale per il calcolo del vantaggio temporale ($GAE$) di ciascun agente.

## 🛠️ Requisiti di Sistema e Prerequisiti

Per garantire la compatibilità con l'API di CARLA 0.9.12 (fornita tramite file `.egg`), il progetto richiede tassativamente:

- **Python 3.7.x** (Consigliato Python 3.7.9 a 64-bit)
- **CARLA Simulator 0.9.12**

---

## 📂 Struttura del Progetto

```text
F1A/
├── config/
│   └── config.py          # Costanti di sistema, configurazione sensori e iperparametri
├── src/
│   ├── __init__.py
│   ├── connection.py      # Gestione della connessione sincrona con il server CARLA
│   ├── environment.py     # Gestione fisica dell'ambiente, spawning e raycasting geometrico
│   ├── main.py              # Entry-point per l'avvio e la gestione del ciclo di simulazione
│   └── models.py          # Classi PyTorch per le reti neurali Actor-Critic
├── .gitignore
├── Monza.npz              # Dataset geometrico dei waypoint del circuito
├── README.md
└── requirements.txt       # Pacchetti necessari per il progetto
```

## 🚀 Installazione e Configurazione

### 1. Download degli Asset e del Simulatore

Il progetto richiede il simulatore CARLA e i file specifici del tracciato di Monza:

- **CARLA 0.9.12:** Scarica la release ufficiale [da questo link](https://github.com/carla-simulator/carla/releases/tag/0.9.12).
- **Pacchetto Mappa di Monza:** Scarica lo zip della mappa (che include il file dei waypoint) [da questo link](https://roar.berkeley.edu/monza-map/).
  - _Nota di posizionamento:_ Estrai il file `Monza.npz` dallo zip e incollalo **all'interno della cartella radice del progetto (`F1A/`)**, esattamente accanto al file `README.md` e alla cartella `src/`.

### 2. Installazione delle dipendenze Python

Apri il terminale nella cartella del progetto (`F1A`) utilizzando l'interprete **Python 3.7** e digita:

```bash
py -3.7 -m pip install -r requirements.txt
```

### 3. Come Eseguire il Progetto

Il progetto richiede l'esecuzione in due passaggi: prima l'avvio del simulatore (Server) e successivamente l'avvio dello script di controllo (Client).

#### Passo 1: Avviare il Server CARLA

1. Apri il terminale del tuo computer (Prompt dei comandi o PowerShell).
2. Naviga fino alla cartella in cui hai installato CARLA 0.9.12.
3. Avvia il simulatore eseguendo il file principale (puoi usare i flag `-quality_level=Low` o `-dx12` se necessario):
   ```bash
   CarlaUE4.exe
   ```
4. Assicurati che il simulatore sia avviato correttamente e rimanga in esecuzione in background.

#### Passo 2: Avviare lo Script Python (Client)

1. Apri un secondo terminale sul tuo computer.
2. Naviga fino alla cartella radice del progetto:
   ```bash
   cd percorso/della/tua/cartella/F1A
   ```
3. Avvia lo script principale assicurandoti di usare l'eseguibile di Python 3.7:
   ```bash
   python src/main.py
   ```
   Nota: Se hai più versioni di Python installate, usa il percorso assoluto o il comando specifico, ad esempio `py -3.7 src/main.py`

📋 Output atteso nel terminale:

```text
📦 File CARLA .egg caricato da: .../PythonAPI/carla/dist/carla-0.9.12-py3.7-win-amd64.egg
🚀 SCRIPT AVVIATO CON SUCCESSO CON PYTHON 3.7!
🗺️ Waypoint di Monza caricati correttamente! (2775 punti trovati, Lunghezza: 5617.05m)
WARNING: Version mismatch detected: You are trying to connect to a simulator that might be incompatible with this API
WARNING: Client API version     = 0.9.12
WARNING: Simulator API version  = 0.9.12-dirty
✅ Client collegato a CARLA con successo! (Modalità sincrona attiva)
🚗 Tutto pronto!
```
