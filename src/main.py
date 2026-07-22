import sys
import os
import torch
import numpy as np

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

import carla
from config.config import MAX_VELOCITY_KMH, NUM_AGENTS, STATE_DIM, GLOBAL_STATE_DIM, ACTION_DIM, MAX_VELOCITY, ROLLOUT_STEPS, LR_ACTOR, LR_CRITIC, GAMMA, LAMBDA, CLIP_EPS, K_EPOCHS, STILL_THRESHOLD_STEPS, MAX_STEPS_PER_EPISODE, COOLDOWN_OVERTAKE_STEPS, OVERTAKE_THRESHOLD_METERS, SAVE_INTERVAL_STEPS
from src.connection import connect_to_carla
from src.environment import waypoint_locations, spawn_initial_vehicles, setup_collision_sensors, get_state, reset_environment, TOTAL_WAYPOINTS, TRACK_LENGTH_METERS, track_distances_cumulative
from src.models import Actor, Critic
from src.buffer import RolloutBuffer
from src.mappo import MAPPO
from src.reward import RewardFunction
from src.logger import TrainingLogger

def apply_initial_velocity(vehicles, speed_kmh=50.0):
    """ Impone una velocità iniziale immediata lungo la direzione del veicolo per sbloccare l'esplorazione """
    speed_mps = speed_kmh / 3.6
    for vehicle in vehicles:
        fwd = vehicle.get_transform().get_forward_vector()
        target_vel = carla.Vector3D(x=fwd.x * speed_mps, y=fwd.y * speed_mps, z=fwd.z * speed_mps)
        vehicle.set_target_velocity(target_vel) # Imposta la velocità target immediatamente

def main():
    # 1. Connessione al simulatore CARLA
    client, world = connect_to_carla()
    print("🚗 Tutto pronto!")

    # Notifica del backend di calcolo (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend_msg = "Abilitata (GPU)" if device.type == "cuda" else "Disabilitata (CPU)"
    print(f"⚙️ Accelerazione PyTorch: {backend_msg}")

    # Recuperiamo la blueprint library dal server CARLA
    blueprint_library = world.get_blueprint_library()

    # 2. Avviamo il primo episodio posizionando correttamente i veicoli a terra tramite reset_environment
    vehicles = []
    collision_sensors = []
    
    vehicles, collision_sensors, collision_types = reset_environment(
        world=world,
        vehicles=vehicles,
        collision_sensors=collision_sensors,
        blueprint_library=blueprint_library
    )

    # 2.1 Spinta iniziale: Applica velocità iniziale al primo avvio (stiamo sul rettilineo di partenza, quindi possiamo dare una spinta iniziale per evitare stalli)
    apply_initial_velocity(vehicles, speed_kmh=40.0)

    # 3. Istanza delle reti Actor-Critic e dell'algoritmo MAPPO
    actor_net = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    critic_net = Critic(global_state_dim=GLOBAL_STATE_DIM, num_agents=NUM_AGENTS).to(device)

    # Caricamento dei pesi pregressi se presenti
    if os.path.exists("actor.pth"):
        actor_net.load_state_dict(torch.load("actor.pth", map_location=device))
        print("✅ Pesi Actor caricati con successo!")

    if os.path.exists("critic.pth"):
        critic_net.load_state_dict(torch.load("critic.pth", map_location=device))
        print("✅ Pesi Critic caricati con successo!")
    
    mappo_agent = MAPPO(
        actor=actor_net,
        critic=critic_net,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        lmbda=LAMBDA,
        eps_clip=CLIP_EPS,
        k_epochs=K_EPOCHS
    )

    # Inizializziamo il buffer di raccolta transizioni
    buffer = RolloutBuffer(
        num_agents=NUM_AGENTS,
        state_dim=STATE_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        action_dim=ACTION_DIM
    )

    # Istanza della funzione di calcolo del reward
    reward_fn = RewardFunction()

    # Istanza del Logger strutturato
    logger = TrainingLogger()

    print(f"🏁 Avvio ciclo di addestramento MAPPO...")

    prev_closest_idx = [None] * NUM_AGENTS # Variabile per salvare l'ultimo indice di waypoint registrato per ciascun agente
    prev_steer_agents = [0.0] * NUM_AGENTS # Memoria dello sterzo precedente per calcolare la penalità dinamica dello sterzo
    episode_id = logger.episode_id # ID dell'episodio
    episode_step = 0 # Contatore degli step all'interno dell'episodio corrente
    done_episode = False # Flag per indicare se l'episodio corrente è terminato (collisione o fine episodio)
    reset_reason = "max_steps" # Di default assume max_steps (verrà sovrascritta in caso di collisione)
    laps_completed = [0] * NUM_AGENTS # contatore dei giri completati per ogni agente
    lap_start_steps = [0] * NUM_AGENTS # Memoria dello step in cui è iniziato l'ultimo giro per ciascun agente
    episode_cumulative_rewards = [0.0, 0.0] # Accumulatore di reward per ciascun agente durante l'episodio
    last_p_loss, last_v_loss = float('nan'), float('nan') # Ultime perdite registrate per l'Actor e il Critic (per logging)
    episode_p_losses = [] # Accumulatore per calcolare la media delle loss per l'Actor sull'intero episodio (per logging)
    episode_v_losses = [] # Accumulatore per calcolare la media delle loss per il Critic sull'intero episodio (per logging)
    still_counter = [0] * NUM_AGENTS # Usato per rilevare se gli agenti sono fermi (o troppo lenti) per troppo tempo (per evitare stalli)

    try:
        # Recupero lo stato iniziale per tutti gli agenti dopo il reset di avvio
        states = []
        for v in vehicles:
            states.append(get_state(v, world))
        states = np.array(states, dtype=np.float32)
        global_state = states.flatten()

        prev_leader = None
        last_overtake_step = -COOLDOWN_OVERTAKE_STEPS
        overtakes_0 = 0
        overtakes_1 = 0
        steps_leading_0 = 0

        # Calcolo dell'ultimo checkpoint salvato per evitare di sovrascrivere i progressi in caso di crash o interruzione
        last_saved_checkpoint = (logger.global_step // SAVE_INTERVAL_STEPS) * SAVE_INTERVAL_STEPS

        while True:
            # 1. Inferenza della Policy
            states_tensor = torch.from_numpy(states).float().to(device)
            global_state_tensor = torch.from_numpy(global_state).float().to(device).unsqueeze(0)

            with torch.no_grad():
                mean, std = actor_net(states_tensor)
                dist = torch.distributions.Normal(mean, std)
                u = dist.rsample()
                actions_tensor = torch.tanh(u)
                log_prob = dist.log_prob(u) - torch.log(1 - actions_tensor.pow(2) + 1e-6)
                log_probs = log_prob.sum(dim=1)
                value = critic_net(global_state_tensor).squeeze(0)

            # 2. Applicazione azioni su CARLA
            actions = actions_tensor.cpu().numpy() # Convertiamo le azioni per CARLA
            for i, vehicle in enumerate(vehicles):
                steer = float(actions[i][0])
                acc_brake = float(actions[i][1])

                if acc_brake >= 0:
                    throttle = acc_brake
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = -acc_brake

                vehicle.apply_control(
                    carla.VehicleControl(
                        steer=steer,
                        throttle=throttle,
                        brake=brake
                    )
                )

            # 3. Avanzamento fisico del server CARLA
            world.tick()

            # 4. Calcolo stato successivo
            next_states = []
            for v in vehicles:
                next_states.append(get_state(v, world))
            next_states = np.array(next_states, dtype=np.float32)
            next_global_state = next_states.flatten()

            rewards = []
            dones = []
            progress_agents = [0.0] * NUM_AGENTS

            # Recupero velocità e angoli fisici denormalizzati
            speed_0 = next_states[0][16] * MAX_VELOCITY
            speed_1 = next_states[1][16] * MAX_VELOCITY
            angle_0 = abs(next_states[0][17] * np.pi)
            angle_1 = abs(next_states[1][17] * np.pi)

            # Estrazione dei dati spaziali
            transforms = [v.get_transform() for v in vehicles]
            locations = [t.location for t in transforms]
            rotations = [t.rotation for t in transforms]

            MIN_VALID_LAP_TIME_S = 40.0

            # 5. Progresso e giri completati per ciascun agente
            for i, vehicle in enumerate(vehicles):
                # Calcolo geometrico del progresso basato sulle nuove coordinate del veicolo
                loc = vehicle.get_transform().location
                wp = waypoint_locations[:, :2]
                distances = np.linalg.norm(wp - np.array([loc.x, loc.y]), axis=1)
                current_closest_idx = np.argmin(distances)

                if prev_closest_idx[i] is None:
                    progress = 0.0
                else:
                    diff = current_closest_idx - prev_closest_idx[i]

                    # Sfasamento traguardo: se il salto è drastico all'indietro, è un nuovo giro
                    if diff < -(TOTAL_WAYPOINTS // 2):
                        laps_completed[i] += 1
                        diff += TOTAL_WAYPOINTS

                        # Calcolo e tracciamento Best Lap Time
                        lap_steps = episode_step - lap_start_steps[i]
                        lap_time_s = lap_steps * 0.05 # 20 FPS -> 0.05s per step
                        if laps_completed[i] > 0 and lap_time_s >= MIN_VALID_LAP_TIME_S:
                            logger.update_lap_time(i, lap_time_s)

                        # Resetta il cronometro per il nuovo giro
                        lap_start_steps[i] = episode_step

                    # Se l'auto va al contrario per errore (marcia indietro drastica oltre il traguardo)
                    elif diff > (TOTAL_WAYPOINTS // 2):
                        laps_completed[i] -= 1
                        diff -= TOTAL_WAYPOINTS

                    progress = diff / TOTAL_WAYPOINTS

                progress_agents[i] = progress
                prev_closest_idx[i] = current_closest_idx

            # Calcoli di telemetria per il Logger
            loc_0 = vehicles[0].get_transform().location
            loc_1 = vehicles[1].get_transform().location
            real_physical_distance = np.sqrt((loc_0.x - loc_1.x)**2 + (loc_0.y - loc_1.y)**2)

            # Chi è davanti?
            idx_0 = prev_closest_idx[0] if prev_closest_idx[0] is not None else 0
            idx_1 = prev_closest_idx[1] if prev_closest_idx[1] is not None else 0

            meters_agent_0 = (laps_completed[0] * TRACK_LENGTH_METERS) + track_distances_cumulative[idx_0]
            meters_agent_1 = (laps_completed[1] * TRACK_LENGTH_METERS) + track_distances_cumulative[idx_1]

            # Calcolo della distanza tra gli agenti lungo l'asfalto
            distance_between_agents = abs(meters_agent_0 - meters_agent_1)

            # True se l'agente 0 è davanti all'agente 1
            is_leading_0 = meters_agent_0 >= meters_agent_1

            # Logica di sorpasso e rilevamento del leader statistico
            lead_distance = meters_agent_0 - meters_agent_1

            if lead_distance > OVERTAKE_THRESHOLD_METERS:
                statistical_leader = 0
            elif lead_distance < -OVERTAKE_THRESHOLD_METERS:
                statistical_leader = 1
            else:
                statistical_leader = prev_leader if prev_leader is not None else (0 if lead_distance >= 0 else 1)

            # Inizializziamo i flag di sorpasso per questo frame
            just_overtook_0 = False
            just_overtook_1 = False

            # Controlla se c'è stato un cambio di leadership valido
            if prev_leader is not None and statistical_leader != prev_leader:
                if episode_step > 50 and speed_0 > 5.0 and speed_1 > 5.0:
                    if episode_step - last_overtake_step > COOLDOWN_OVERTAKE_STEPS:  # Evita oscillazioni rapide
                        if statistical_leader == 0:
                            just_overtook_0 = True
                            overtakes_0 += 1
                        else:
                            just_overtook_1 = True
                            overtakes_1 += 1
                        last_overtake_step = episode_step

            prev_leader = statistical_leader
            if statistical_leader == 0:
                steps_leading_0 += 1

            # Calcolo reward per ciascun agente e gestione collisioni
            for i, vehicle in enumerate(vehicles):
                # Estraiamo le metriche dallo stato successivo
                speed_norm = next_states[i][16] # Velocità normalizzata (0-1)
                speed_kmh = speed_norm * MAX_VELOCITY_KMH
                angle_norm = next_states[i][17] # Angolo normalizzato (0-1)
                collision = (collision_types[i] is not None)

                # Identifica i flag specifici per l'agente corrente (i)
                is_current_agent_leader = (statistical_leader == i)
                just_overtook_current = just_overtook_0 if i == 0 else just_overtook_1

                # Rilevamento se l'agente è fermo o troppo lento per troppo tempo
                if speed_kmh < 7.0:
                    still_counter[i] += 1
                else:
                    still_counter[i] = 0

                collision = (collision_types[i] is not None)

                if episode_step > 30:
                    if collision:
                        reset_reason = collision_types[i]
                        collision_types[i] = None
                        done_episode = True
                    elif still_counter[i] > STILL_THRESHOLD_STEPS:
                        done_episode = True
                        if reset_reason not in ["wall_collision", "car_collision"]:
                            reset_reason = "still"
                    elif episode_step >= MAX_STEPS_PER_EPISODE:
                        done_episode = True
                        if reset_reason not in ["wall_collision", "car_collision", "still"]:
                            reset_reason = "max_steps"
                else:
                    # Svuota i sensori durante i primi 30 passi per ripulire i micro-rimbalzi dello spawn
                    collision_types[i] = None

                # Risoluzione speculare delle variabili spaziali locali per l'agente corrente (i)
                loc_self = locations[i]
                rot_self = rotations[i]
                loc_target = locations[1] if i == 0 else locations[0]

                # Calcolo del reward
                reward = reward_fn.calculate_reward(
                    progress=progress_agents[i],
                    angle_norm=angle_norm,
                    speed_norm=speed_norm,
                    collision=collision,
                    is_leading=is_current_agent_leader,
                    distance_between_agents=distance_between_agents,
                    real_physical_distance=real_physical_distance,
                    current_steer=actions[i][0],
                    prev_steer=prev_steer_agents[i],
                    just_overtook=just_overtook_current,
                    loc_0=loc_self,
                    loc_1=loc_target,
                    rot_0=rot_self,
                    episode_step=episode_step
                )

                rewards.append(reward)
                dones.append(collision)

                prev_steer_agents[i] = actions[i][0]

            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)

            # Accumula la reward del frame corrente
            episode_cumulative_rewards[0] += float(rewards[0])
            episode_cumulative_rewards[1] += float(rewards[1])

            speed_kmh_0 = next_states[0][16] * MAX_VELOCITY_KMH
            speed_kmh_1 = next_states[1][16] * MAX_VELOCITY_KMH

            # Calcolo distanze in metri e progresso % per il logger
            current_distances_m = [meters_agent_0, meters_agent_1]
            current_progress_pcts = [
                (meters_agent_0 / TRACK_LENGTH_METERS) * 100.0,
                (meters_agent_1 / TRACK_LENGTH_METERS) * 100.0
            ]

            # Inviamo i dati allo step recorder
            logger.record_step(
                speeds=[speed_kmh_0, speed_kmh_1],
                angles=[angle_0, angle_1],
                distance_between_agents=distance_between_agents,
                real_physical_distance=real_physical_distance,
                is_leading_0=is_leading_0,
                rewards=rewards.tolist(),
                current_distances_m=current_distances_m,
                current_progress_pcts=current_progress_pcts
            )

            # Controlliamo se è il momento di salvare un checkpoint automatico
            if logger.global_step >= last_saved_checkpoint + SAVE_INTERVAL_STEPS:
                end_step = logger.global_step

                # Definiamo il percorso della cartella dinamica: pth-history/checkpoint-{end_step}_steps
                history_dir = os.path.join("pth-history", f"checkpoint-{end_step}_steps")
                os.makedirs(history_dir, exist_ok=True)

                # Salvataggio dei file all'interno della cartella specifica
                actor_history_path = os.path.join(history_dir, "actor.pth")
                critic_history_path = os.path.join(history_dir, "critic.pth")

                torch.save(actor_net.state_dict(), actor_history_path)
                torch.save(critic_net.state_dict(), critic_history_path)

                print(f"💾 Checkpoint automatico salvato in: {history_dir} (Global Step: {end_step})")
                last_saved_checkpoint = end_step

            # 6. Salvataggio nel buffer
            buffer.store(
                states=states,
                global_state=global_state,
                actions=actions,
                log_probs=log_probs.cpu().numpy(),
                rewards=rewards,
                dones=dones,
                values=value.cpu().numpy()
            )

            # Avanzamento di stato
            states = next_states
            global_state = next_global_state
            episode_step += 1

            # Gestione fine episodio e reset dell'ambiente
            if done_episode:
                print(f"🚨 Episodio finito. Step episodio: {episode_step}. Motivo: {reset_reason}")

                # Se abbiamo accumulato abbastanza dati totali, facciamo l'update e prendiamo i dati di perdita
                if len(buffer) >= ROLLOUT_STEPS:
                    print(f"🎯 Rollout completo ({len(buffer)} step). Ottimizzazione MAPPO...")
                    last_p_loss, last_v_loss = mappo_agent.update(buffer, global_state, device)
                    episode_p_losses.append(last_p_loss)
                    episode_v_losses.append(last_v_loss)

                # Calcola la media delle loss dell'intero episodio
                avg_p_loss = np.mean(episode_p_losses) if episode_p_losses else float('nan')
                avg_v_loss = np.mean(episode_v_losses) if episode_v_losses else float('nan')

                # Salva i dati su CSV chiamando il logger
                logger.log_episode_end(
                    laps_completed=laps_completed,
                    overtakes=[overtakes_0, overtakes_1],
                    losses=(avg_p_loss, avg_v_loss),
                    reason=reset_reason
                )

                episode_id += 1

                # Reset dell'ambiente
                vehicles, collision_sensors, collision_types = reset_environment(
                    world=world,
                    vehicles=vehicles,
                    collision_sensors=collision_sensors,
                    blueprint_library=blueprint_library
                )

                # Applica la velocità di partenza a 40 km/h per il nuovo episodio
                apply_initial_velocity(vehicles, speed_kmh=40.0)

                # Riavvia le variabili di stato post-reset
                states = []
                for v in vehicles:
                    states.append(get_state(v, world))
                states = np.array(states, dtype=np.float32)
                global_state = states.flatten()

                episode_step = 0
                done_episode = False
                reset_reason = "max_steps"
                prev_steer_agents = [0.0] * NUM_AGENTS
                prev_closest_idx = [None] * NUM_AGENTS
                laps_completed = [0] * NUM_AGENTS
                lap_start_steps = [0] * NUM_AGENTS
                episode_cumulative_rewards = [0.0, 0.0]
                still_counter = [0] * NUM_AGENTS
                overtakes_0 = 0
                overtakes_1 = 0
                steps_leading_0 = 0
                prev_leader = None
                last_overtake_step = -COOLDOWN_OVERTAKE_STEPS
                last_p_loss, last_v_loss = float('nan'), float('nan')
                episode_p_losses = []
                episode_v_losses = []

                continue

            # Se l'episodio prosegue normalmente ma raggiungiamo la soglia di rollout
            if len(buffer) >= ROLLOUT_STEPS:
                print(f"🎯 Rollout completo ({len(buffer)} step). Ottimizzazione MAPPO...")
                last_p_loss, last_v_loss = mappo_agent.update(buffer, global_state, device)
                episode_p_losses.append(last_p_loss)
                episode_v_losses.append(last_v_loss)

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