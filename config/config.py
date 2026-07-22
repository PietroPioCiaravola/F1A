# Parametri
NUM_AGENTS = 2 # Numero di agenti (auto) nel simulatore
DEAD_ZONE = 0.05 # Zona morta per l'angolo di sterzo (in radianti) entro la quale non viene applicata penalità

# Tracciato
WAYPOINT_FILE = "Monza.npz" # File dei waypoint del tracciato di Monza (in formato .npz)
TRUE_START_INDEX = 2505 # Indice del waypoint di partenza reale (per il tracciato di Monza, questo è il punto di partenza corretto)

# Configurazione fisica e limiti
MAX_VELOCITY = 70.00  # Velocità massima di riferimento (m/s) per la normalizzazione (~252 km/h). Velocità massima raggiungibile da una tesla model 3 su CARLA: 71.58 m/s (257.7 km/h)
MAX_VELOCITY_KMH = MAX_VELOCITY * 3.6  # Velocità massima in km/h
STILL_THRESHOLD_STEPS = 300  # Numero di step consecutivi a velocità quasi nulla prima del reset per "still", 300 X 0.05 = 15 secondi
MAX_STEPS_PER_EPISODE = 2000  # Limite massimo di step prima del reset forzato dell'ambiente
SAVE_INTERVAL_STEPS = 50000 # Salva i checkpoint ogni SAVE_INTERVAL_STEPS step globali per evitare di perdere progressi in caso di crash o interruzione + Permettere di riprendere l'addestramento da un punto intermedio
COOLDOWN_OVERTAKE_STEPS = 100  # Numero di step di cooldown dopo un sorpasso prima di poter effettuare un nuovo sorpasso (100 X 0.05 = 5 secondi)
OVERTAKE_THRESHOLD_METERS = 5.5 # Distanza minima tra due agenti per considerare un sorpasso valido (in metri).

# Configurazione dei Raycast
ANGLES = [
    0,                     # Dritto davanti
    -10, 10,               # Fascio strettissimo centrale (staccate lunghe)
    -20, 20,               # Uscita dalle curve veloci
    -35, 35,               # Diagonale anteriore (inserimento)
    -50, 50,               # Approccio cordoli
    -70, 70,               # Visione laterale avanzata
    -90, 90,               # Laterale puro (Vicinanza rivale nel sorpasso)
    -135, 135,             # Diagonale posteriore (scia)
    180                    # Retronebbia
]

RAY_LENGTHS = [
    110,                    # 0°:
    100, 100,              # ±10°
    85, 85,                # ±20°
    45, 45,                # ±35°
    40, 40,                # ±50°
    35, 35,                # ±70°
    30, 30,                # ±90°:
    20, 20,                # ±135°
    15                     # 180°
]

# Dimensioni delle reti neurali
STATE_DIM = 18          # 16 raggi + velocità norm + angolo norm
ACTION_DIM = 2          # Sterzo, Acceleratore/Freno
GLOBAL_STATE_DIM = STATE_DIM * NUM_AGENTS  # Stato globale per il Critic centralizzato

# Iperparametri MAPPO
ROLLOUT_STEPS = 512     # Quanti step accumulare nel buffer prima di aggiornare le reti
LR_ACTOR = 3e-4         # Learning rate dell'Actor
LR_CRITIC = 3e-4        # Learning rate del Critic
GAMMA = 0.99            # Fattore di sconto temporale
LAMBDA = 0.95           # Parametro GAE (Generalized Advantage Estimation)
CLIP_EPS = 0.2          # Parametro di clipping per PPO
K_EPOCHS = 5            # Numero di epoche di ottimizzazione per aggiornamento

# Pesi Reward
W_PROGRESS = 20.0 # Peso del progresso lungo la pista
W_SPEED = 1.0 # Peso della velocità
W_DIRECTION = 1.5 # Peso della direzione (angolo di sterzo)
W_COLLISION = 10.0 # Peso della collisione
W_COMPETITION = 1.0 # Peso della competizione