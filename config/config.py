# Parametri
NUM_AGENTS = 2 # Numero di agenti (auto) nel simulatore

# Tracciato
WAYPOINT_FILE = "Monza.npz" # File dei waypoint del tracciato di Monza (in formato .npz)
TRUE_START_INDEX = 2505 # Indice del waypoint di partenza reale (per il tracciato di Monza, questo è il punto di partenza corretto)

# Configurazione fisica
MAX_VELOCITY = 40.0  # Velocità massima di riferimento (m/s) per la normalizzazione (~144 km/h)

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
    65,                    # 0°:
    60, 60,                # ±10°
    55, 55,                # ±20°
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