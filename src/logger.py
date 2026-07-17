import os
import csv
import numpy as np

class TrainingLogger:
    def __init__(self, csv_path="training_log.csv", num_agents=2):
        self.csv_path = csv_path
        self.num_agents = num_agents
        
        # Intestazione del CSV con le metriche da registrare
        self.headers = [
            "episode_id", "step", "reward_mean", "reward_total", "episode_length",
            "avg_speed_0", "avg_speed_1", "avg_angle_0", "avg_angle_1",
            "laps_0", "laps_1", "min_distance", "avg_distance", "pct_leader_0", "pct_leader_1",
            "policy_loss", "value_loss", "reason"
        ]
        
        self.episode_id = 0
        self.global_step = 0
        self._init_csv_file()
        self.reset_episode_stats()

    def _init_csv_file(self):
        """Inizializza il file CSV o riprende l'addestramento esistente."""
        if os.path.isfile(self.csv_path):
            try:
                with open(self.csv_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1: # Se c'è almeno un episodio salvato (La prima riga è l'intestazione)
                        last_line = lines[-1].strip().split(",")
                        self.episode_id = int(last_line[0]) + 1 # Il prossimo episodio sarà +1
                        self.global_step = int(last_line[1])    # Riprendiamo lo step cumulativo
                        print(f"🔄 Logger caricato! Riprendo dall'Episodio: {self.episode_id} (Step: {self.global_step})")
                        return
            except Exception as e:
                print(f"⚠️ Impossibile leggere il CSV ({e}). Riparto da zero.")
        
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def reset_episode_stats(self):
        """Azzera gli accumulatori per il nuovo episodio."""
        self.episode_step = 0
        self.steps_leading_0 = 0
        self.min_distance_recorded = 999.0
        self.episode_stats = {
            "speeds_0": [], "speeds_1": [],
            "angles_0": [], "angles_1": [],
            "inter_agent_distances": [],
            "rewards": []
        }

    def record_step(self, speeds, angles, distance_between_agents, real_physical_distance, is_leading_0, rewards):
        """Registra le metriche istantanee del singolo frame."""
        self.episode_step += 1
        self.global_step += 1
        
        # Accumula velocità e allineamenti
        self.episode_stats["speeds_0"].append(speeds[0])
        self.episode_stats["speeds_1"].append(speeds[1])
        self.episode_stats["angles_0"].append(angles[0])
        self.episode_stats["angles_1"].append(angles[1])
        
        # Accumula le distanze e i reward
        self.episode_stats["inter_agent_distances"].append(distance_between_agents)
        self.episode_stats["rewards"].extend(rewards)

        # Calcolo della distanza minima registrata (dopo un piccolo tempo di grazia per lo spawn)
        if self.episode_step > 30:
            if distance_between_agents > real_physical_distance:
                # C'è una curva a separarli: usiamo la distanza topologica sulla pista
                if distance_between_agents < self.min_distance_recorded:
                    self.min_distance_recorded = distance_between_agents
            else:
                # Sono sullo stesso rettilineo o vicini: usiamo la precisione euclidea
                if real_physical_distance < self.min_distance_recorded:
                    self.min_distance_recorded = real_physical_distance

        # Monitoraggio del leader
        if is_leading_0:
            self.steps_leading_0 += 1

    def log_episode_end(self, laps_completed, losses, reason):
        """Calcola le medie, scrive sul CSV e incrementa l'ID dell'episodio."""
        ep_mean_reward = np.mean(self.episode_stats["rewards"]) if self.episode_stats["rewards"] else 0.0
        ep_total_reward = np.sum(self.episode_stats["rewards"]) if self.episode_stats["rewards"] else 0.0
        avg_speed_0 = np.mean(self.episode_stats["speeds_0"]) if self.episode_stats["speeds_0"] else 0.0
        avg_speed_1 = np.mean(self.episode_stats["speeds_1"]) if self.episode_stats["speeds_1"] else 0.0
        avg_angle_0 = np.mean(self.episode_stats["angles_0"]) if self.episode_stats["angles_0"] else 0.0
        avg_angle_1 = np.mean(self.episode_stats["angles_1"]) if self.episode_stats["angles_1"] else 0.0
        avg_distance = np.mean(self.episode_stats["inter_agent_distances"]) if self.episode_stats["inter_agent_distances"] else 0.0
        pct_leading_0 = (self.steps_leading_0 / self.episode_step * 100) if self.episode_step > 0 else 0.0
        pct_leading_1 = 100.0 - pct_leading_0 if self.episode_step > 0 else 0.0

        final_min_dist = f"{self.min_distance_recorded:.2f}" if self.min_distance_recorded != 999.0 else "NaN"
        policy_loss, value_loss = losses

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_id, self.global_step, f"{ep_mean_reward:.4f}", f"{ep_total_reward:.2f}", self.episode_step,
                f"{avg_speed_0:.2f}", f"{avg_speed_1:.2f}", f"{avg_angle_0:.3f}", f"{avg_angle_1:.3f}",
                laps_completed[0], laps_completed[1], final_min_dist, f"{avg_distance:.2f}", f"{pct_leading_0:.1f}", f"{pct_leading_1:.1f}",
                f"{policy_loss:.5f}", f"{value_loss:.5f}", reason
            ])

        print(f"📊 Episodio {self.episode_id} loggato con successo. Reward media: {ep_mean_reward:.4f}")
        
        self.episode_id += 1
        self.reset_episode_stats()