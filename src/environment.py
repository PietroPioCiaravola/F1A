import carla
import math
import os
import numpy as np
from config.config import *
from src.reward import RewardFunction

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

def spawn_initial_vehicles(world, waypoint_locations, num_agents=2, frame_w=800, frame_h=600):
    """
    Spawna i veicoli affiancati sul primo waypoint.
    Ritorna la lista dei veicoli spawnati.
    """
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # Calcolo geometrico per lo spawn affiancato
    wp_zero = waypoint_locations[0]
    wp_one = waypoint_locations[1]

    # Calcoliamo la direzione e l'angolo esatto dal punto 0 al punto 1 della pista
    yaw_rad = math.atan2(wp_one[1] - wp_zero[1], wp_one[0] - wp_zero[0])
    yaw_deg = math.degrees(yaw_rad)
    rotation_auto = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)

    right_vector = carla.Vector3D(x=-math.sin(yaw_rad), y=math.cos(yaw_rad), z=0)
    VEHICLE_WIDTH = 2.0
    LATERAL_MARGIN = 1.2
    offset = VEHICLE_WIDTH / 2 + LATERAL_MARGIN

    vehicles = []

    for i in range(num_agents):
        lateral = offset if i % 2 == 0 else -offset

        spawn_transform = carla.Transform(
            carla.Location(
                x=wp_zero[0] + right_vector.x * lateral,
                y=wp_zero[1] + right_vector.y * lateral,
                z=wp_zero[2] + 0.1
            ),
            rotation_auto
        )

        bp_locale = blueprint_library.filter('vehicle.tesla.model3')[0]
        if i == 0:
            bp_locale.set_attribute('color', '255,0,0')    # Rosso Corsa
        else:
            bp_locale.set_attribute('color', '0,242,255')  # Turchese/Cyan

        vehicle = world.try_spawn_actor(bp_locale, spawn_transform)

        if vehicle:
            vehicles.append(vehicle)
        else:
            # Se CARLA fallisce lo spawn, solleva un'eccezione per indicare il problema
            raise RuntimeError(f"Errore critico: Impossibile spawnare il veicolo {i} nella posizione indicata.")

    return vehicles

def setup_collision_sensors(world, vehicles):
    """
    Configura e aggancia un sensore di collisione a ciascun veicolo.
    Ritorna la lista dei sensori creati e una lista condivisa per tracciare lo stato delle collisioni.
    """
    collision_types = [None] * len(vehicles)
    collision_sensors = []
    blueprint_library = world.get_blueprint_library()
    collision_bp = blueprint_library.find('sensor.other.collision')

    for i, vehicle in enumerate(vehicles):
        sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle
        )

        def make_callback(index):
            def callback(event):
                other_actor = event.other_actor
                if other_actor and "vehicle" in other_actor.type_id:
                    collision_types[index] = "car_collision"
                else:
                    collision_types[index] = "wall_collision"
            return callback

        sensor.listen(make_callback(i))
        collision_sensors.append(sensor)

    return collision_sensors, collision_types

def get_state(vehicle, world):
    """Calcola lo stato dell'agente includendo i 16 raycast + la velocità e l'allineamento pista normalizzati."""
    transform = vehicle.get_transform()
    location = transform.location
    yaw = math.radians(transform.rotation.yaw)

    start = carla.Location(
        x=location.x,
        y=location.y,
        z=location.z + 1.0
    )

    bbox = vehicle.bounding_box
    extent_x = bbox.extent.x
    extent_y = bbox.extent.y

    ray_distances = []

    for angle, ray_length in zip(ANGLES, RAY_LENGTHS):
        angle_rad = math.radians(angle)
        offset = math.sqrt(
            (extent_x * math.cos(angle_rad))**2 +
            (extent_y * math.sin(angle_rad))**2
        )
        ray_yaw = yaw + angle_rad

        end = carla.Location(
            x=start.x + ray_length * math.cos(ray_yaw),
            y=start.y + ray_length * math.sin(ray_yaw),
            z=start.z
        )

        hits = world.cast_ray(start, end)
        distance = ray_length
        hits_sorted = sorted(hits, key=lambda h: start.distance(h.location))

        for hit in hits_sorted:
            d = start.distance(hit.location)
            if d > (offset + 0.1):
                distance = d
                break

        distance_corrected = max(0.0, distance - offset)
        normalized = distance_corrected / ray_length
        ray_distances.append(normalized)

    # Calcolo Velocità
    velocity = vehicle.get_velocity()
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    speed_norm = np.clip(speed / MAX_VELOCITY, 0.0, 1.0)

    # Angolo relativo rispetto ai waypoint della pista
    wp = waypoint_locations[:, :2]
    distances = np.linalg.norm(
        wp - np.array([location.x, location.y]),
        axis=1
    )
    closest_idx = np.argmin(distances)

    # Recuperiamo la direzione in cui guarda l'auto
    v_forward = transform.get_forward_vector()
    # Recuperiamo la direzione reale della pista usando il tracciato dei waypoint
    next_idx = (closest_idx + 1) % len(waypoint_locations)
    wp_now = waypoint_locations[closest_idx]
    wp_next = waypoint_locations[next_idx]
    # Vettore direzione pista
    track_dir_x = wp_next[0] - wp_now[0]
    track_dir_y = wp_next[1] - wp_now[1]
    # Calcoliamo l'angolo relativo tra l'auto e la pista
    angle_vehicle = math.atan2(v_forward.y, v_forward.x)
    angle_track = math.atan2(track_dir_y, track_dir_x)
    
    angle_to_track = angle_track - angle_vehicle
    # Normalizziamo tra -pi e +pi
    angle_to_track = (angle_to_track + math.pi) % (2 * math.pi) - math.pi
    angle_norm = angle_to_track / math.pi

    state = np.concatenate([
        ray_distances,
        [speed_norm],
        [angle_norm]
    ])

    return state

def reset_environment(world, vehicles, collision_sensors, blueprint_library):
    """
    Esegue il reset completo dell'ambiente: pulizia, spawning randomizzato dei veicoli sulla linea di partenza (swap 50%)
    e stabilizzazione fisica a terra.
    La funzione allo stato attuale gestisce lo spawn di due veicoli affiancati sulla linea di partenza.
    """
    # 1. Distruggi veicoli in sicurezza
    for v in vehicles:
        try:
            v.destroy()
        except:
            pass
    vehicles.clear()

    # 2. Distruggi sensori in sicurezza
    for s in collision_sensors:
        try:
            s.destroy()
        except:
            pass
    collision_sensors.clear()

    # Forza CARLA a elaborare la distruzione prima di spawnare i nuovi attori
    world.tick()

    # 3. Reset stati locali delle collisioni
    collision_types = [None] * NUM_AGENTS

    # 4. Calcoli geometrici per il respawn affiancato sul traguardo
    wp_zero = waypoint_locations[0]
    wp_one = waypoint_locations[1]

    yaw_rad = math.atan2(wp_one[1] - wp_zero[1], wp_one[0] - wp_zero[0])
    yaw_deg = math.degrees(yaw_rad)
    rotation_auto = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)

    right_vector = carla.Vector3D(
        x=-math.sin(yaw_rad),
        y=math.cos(yaw_rad),
        z=0
    )

    # 🎯 TESTA O CROCE: 50% di probabilità di invertire i lati degli agenti
    swap_sides = np.random.rand() > 0.5
    VEHICLE_WIDTH = 2.0
    LATERAL_MARGIN = 1.2

    for i in range(NUM_AGENTS):
        # Calcoliamo la distanza standard dal centro della pista
        offset = VEHICLE_WIDTH / 2 + LATERAL_MARGIN

        # Se swap_sides è True, invertiamo la posizione dei veicoli
        if swap_sides:
            lateral = -offset if i % 2 == 0 else offset
        else:
            lateral = offset if i % 2 == 0 else -offset

        spawn_transform = carla.Transform(
            carla.Location(
                x=wp_zero[0] + right_vector.x * lateral,
                y=wp_zero[1] + right_vector.y * lateral,
                z=wp_zero[2] + 0.1
            ),
            rotation_auto
        )

        # Spawn del veicolo con colore differente per distinguere i due agenti
        bp_locale = blueprint_library.filter('vehicle.tesla.model3')[0]
        if i == 0:
            bp_locale.set_attribute('color', '255,0,0')    # Rosso Corsa
        else:
            bp_locale.set_attribute('color', '0,242,255')  # Turchese/Cyan

        vehicle = world.try_spawn_actor(bp_locale, spawn_transform)
        if vehicle:
            vehicles.append(vehicle)

    # 5. Ricrea sensori di collisione
    collision_bp = blueprint_library.find('sensor.other.collision')
    for i, vehicle in enumerate(vehicles):
        sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle
        )

        def make_callback(index):
            def callback(event):
                other_actor = event.other_actor
                if other_actor and "vehicle" in other_actor.type_id:
                    collision_types[index] = "car_collision"
                else:
                    collision_types[index] = "wall_collision"
            return callback

        sensor.listen(make_callback(i))
        collision_sensors.append(sensor)

    # 6. Stabilizzazione Fisica (Tiriamo il freno a mano per far cadere l'auto a terra)
    for vehicle in vehicles:
        vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0, hand_brake=True))
    
    # Facciamo avanzare il simulatore a vuoto senza registrare traiettorie
    TICKS_TO_STABILIZE = 15  # 15 tick * 0.05s = 0.75 secondi di stabilizzazione fisica a terra
    for _ in range(TICKS_TO_STABILIZE):
        world.tick()
    
    # Rilasciamo i freni per iniziare l'episodio
    for vehicle in vehicles:
        controllo_pulito = carla.VehicleControl(
            hand_brake=False, 
            brake=0.0, 
            throttle=0.0, 
            steer=0.0,
            manual_gear_shift=False # Forza il cambio automatico di CARLA a inserire la "Drive"
        )
        vehicle.apply_control(controllo_pulito)
    
    # Facciamo un singolo tick a vuoto per applicare lo sblocco prima di ridare il controllo all'IA
    world.tick()

    # Svuota le collisioni registrate accidentalmente durante la caduta iniziale
    for index in range(NUM_AGENTS):
        collision_types[index] = None

    return vehicles, collision_sensors, collision_types