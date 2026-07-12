import os
import numpy as np
from config.config import *

waypoint_locations = None
TOTAL_WAYPOINTS = 0
TRACK_LENGTH_METERS = 0.0

src_dir = os.path.dirname(os.path.abspath(__file__))
f1a_root_dir = os.path.dirname(src_dir)
absolute_waypoint_path = os.path.join(f1a_root_dir, WAYPOINT_FILE)

# --- Inizializzazione Tracciato ---
if os.path.exists(absolute_waypoint_path):
    data = np.load(absolute_waypoint_path)
    waypoint_locations = data['locations']
    if TRUE_START_INDEX != 0:
        # Ruota l'array dei waypoint in modo che il punto di partenza sia all'indice corretto
        waypoint_locations = np.roll(waypoint_locations, -TRUE_START_INDEX, axis=0)
    # Inverti l'asse Y per allineare il sistema di coordinate con quello del simulatore
    waypoint_locations[:, 1] = -waypoint_locations[:, 1]

    # Calcola la lunghezza totale del tracciato
    TOTAL_WAYPOINTS = len(waypoint_locations)
    wp_xy = waypoint_locations[:, :2]
    segment_distances = np.linalg.norm(wp_xy - np.roll(wp_xy, -1, axis=0), axis=1)
    TRACK_LENGTH_METERS = np.sum(segment_distances)

    print(f"🗺️ Waypoint di Monza caricati correttamente! ({TOTAL_WAYPOINTS} punti trovati, Lunghezza: {TRACK_LENGTH_METERS:.2f}m)")
else:
    print(f"❌ Errore fatale: Impossibile trovare il file dei waypoint in: {absolute_waypoint_path}")