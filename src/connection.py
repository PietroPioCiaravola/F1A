import carla
from config.config import *

def connect_to_carla():
    """Inizializza la connessione con il simulatore CARLA."""
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    world = client.get_world()

    # Configurazione modalità sincrona
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    print("✅ Client collegato a CARLA con successo! (Modalità sincrona attiva)")
    
    return client, world