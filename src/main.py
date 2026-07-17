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
from config.config import NUM_AGENTS, STATE_DIM, GLOBAL_STATE_DIM, ACTION_DIM, MAX_VELOCITY, ROLLOUT_STEPS, LR_ACTOR, LR_CRITIC, GAMMA, LAMBDA, CLIP_EPS, K_EPOCHS
from src.connection import connect_to_carla
from src.environment import waypoint_locations, spawn_initial_vehicles, setup_collision_sensors, get_state, reset_environment, TOTAL_WAYPOINTS, TRACK_LENGTH_METERS
from src.models import Actor, Critic
from src.buffer import RolloutBuffer
from src.mappo import MAPPO
from src.reward import RewardFunction

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

    # 3. Istanza delle reti Actor-Critic e dell'algoritmo MAPPO
    actor_net = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    critic_net = Critic(global_state_dim=GLOBAL_STATE_DIM, num_agents=NUM_AGENTS).to(device)
    
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

    print(f"🏁 Avvio ciclo di addestramento MAPPO...")

    prev_closest_idx = [None] * NUM_AGENTS # Variabile per salvare l'ultimo indice di waypoint registrato per ciascun agente
    prev_steer_agents = [0.0] * NUM_AGENTS # Memoria dello sterzo precedente per calcolare la penalità dinamica dello sterzo
    episode_id = 0 # ID dell'episodio
    episode_step = 0 # Contatore degli step all'interno dell'episodio corrente
    done_episode = False # Flag per indicare se l'episodio corrente è terminato (collisione o fine episodio)

    try:
        # Recupero lo stato iniziale per tutti gli agenti dopo il reset di avvio
        states = []
        for v in vehicles:
            states.append(get_state(v, world))
        states = np.array(states, dtype=np.float32)
        global_state = states.flatten()

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
            any_collision = False

            # 5. Calcolo reward per ciascun agente e getione collisioni
            for i, vehicle in enumerate(vehicles):
                # Estraiamo le metriche dallo stato successivo
                speed_norm = next_states[i][16] # Velocità normalizzata (0-1)
                angle_norm = next_states[i][17] # Angolo normalizzato (0-1)

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
                        diff += TOTAL_WAYPOINTS
                    # Se l'auto va al contrario per errore (marcia indietro drastica oltre il traguardo)
                    elif diff > (TOTAL_WAYPOINTS // 2):
                        diff -= TOTAL_WAYPOINTS

                    progress = diff / TOTAL_WAYPOINTS

                prev_closest_idx[i] = current_closest_idx

                collision = (collision_types[i] is not None)
                if collision:
                    collision_types[i] = None
                    any_collision = True

                # Calcolo del reward
                reward = reward_fn.calculate_reward(
                    progress=progress,
                    angle_norm=angle_norm,
                    speed_norm=speed_norm,
                    collision=collision,
                    acc_brake=actions[i][1],
                    current_steer=actions[i][0],
                    prev_steer=prev_steer_agents[i]
                )
                rewards.append(reward)
                dones.append(collision)

                prev_steer_agents[i] = actions[i][0]

            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)

            # Condizione di fine episodio se c'è una collisione
            if any_collision:
                done_episode = True

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
                print(f"🚨 Episodio finito. Step episodio: {episode_step}")

                # Se abbiamo accumulato abbastanza dati totali, facciamo l'update
                if len(buffer) >= ROLLOUT_STEPS:
                    print(f"🎯 Rollout completo ({len(buffer)} step). Ottimizzazione MAPPO...")
                    mappo_agent.update(buffer, global_state, device)

                episode_id += 1

                # Reset dell'ambiente
                vehicles, collision_sensors, collision_types = reset_environment(
                    world=world,
                    vehicles=vehicles,
                    collision_sensors=collision_sensors,
                    blueprint_library=blueprint_library
                )

                # Riavvia le variabili di stato post-reset
                states = []
                for v in vehicles:
                    states.append(get_state(v, world))
                states = np.array(states, dtype=np.float32)
                global_state = states.flatten()

                episode_step = 0
                done_episode = False
                any_collision = False
                prev_steer_agents = [0.0] * NUM_AGENTS
                prev_closest_idx = [None] * NUM_AGENTS

                continue

            # Se l'episodio prosegue normalmente ma raggiungiamo la soglia di rollout
            if len(buffer) >= ROLLOUT_STEPS:
                print(f"🎯 Rollout completo ({len(buffer)} step). Ottimizzazione MAPPO...")
                mappo_agent.update(buffer, global_state, device)

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