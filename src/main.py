import sys
import os

# --- Caricamento dinamico CARLA Egg ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_monza = None
while True:
    if os.path.basename(current_dir) == "Monza":
        root_monza = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        break
    current_dir = parent

if root_monza:
    EGG_PATH = os.path.join(root_monza, "PythonAPI", "carla", "dist", "carla-0.9.12-py3.7-win-amd64.egg")
    if os.path.exists(EGG_PATH):
        sys.path.insert(0, EGG_PATH)
        print(f"📦 File CARLA .egg caricato da: {EGG_PATH}")
    else:
        print(f"⚠️ Attenzione: File .egg non trovato in {EGG_PATH}")

f1a_root = os.path.dirname(os.path.abspath(__file__)) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "F1A" else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f1a_root)

# Controlla se la versione principale è 3 e la secondaria è 7s
if sys.version_info.major != 3 or sys.version_info.minor != 7:
    sys.exit(f"❌ Errore: Questo script richiede rigorosamente Python 3.7 per caricare l'.egg di CARLA 0.9.12.\n"
            f"Attualmente stai usando Python {sys.version_info.major}.{sys.version_info.minor}.\n")

print("🚀 SCRIPT AVVIATO CON SUCCESSO CON PYTHON 3.7!")

from config.config import NUM_AGENTS
from src.connection import connect_to_carla
from src.environment import waypoint_locations, spawn_initial_vehicles, setup_collision_sensors, get_state

def main():
    # 1. Connessione al simulatore
    client, world = connect_to_carla()
    print("🚗 Tutto pronto!")

    # 2. Spawn dei veicoli all'avvio dello script
    vehicles = spawn_initial_vehicles(
        world=world,
        waypoint_locations=waypoint_locations,
        num_agents=NUM_AGENTS
    )

    # 3. Setup dei sensori di collisione agganciati alle vetture
    collision_sensors, collision_types = setup_collision_sensors(world, vehicles)

    try:
        while True:
            world.tick()
    finally:
        # 1. Ripristiniamo la modalità asincrona per evitare che CARLA si congeli.
        try:
            print("🔄 Ripristino modalità asincrona di CARLA...")
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except:
            pass

        # 2. Distruggiamo i sensori
        for sensor in collision_sensors:
            try: sensor.destroy()
            except: pass

        # 3. Distruggiamo i veicoli
        for vehicle in vehicles:
            try: vehicle.destroy()
            except: pass

        print("🧹 Pulizia completata. Script terminato in sicurezza.")

if __name__ == "__main__":
    main()