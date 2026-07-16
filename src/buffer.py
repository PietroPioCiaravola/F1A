import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, num_agents, state_dim, global_state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim

        self.reset()

    def reset(self):
        """Svuota il buffer all'inizio di ogni ciclo di raccolta dati."""
        self.states = []         # Stati locali di ciascun agente
        self.global_states = []  # Stati globali (unione degli stati locali per il Critic)
        self.actions = []        # Azioni scelte (sterzo, acc/freno)
        self.log_probs = []      # Log-probabilità delle azioni
        self.rewards = []        # Reward ricevute
        self.dones = []          # Flag di fine episodio
        self.values = []         # Stime dei valori calcolate dal Critic

    def store(self, states, global_state, actions, log_probs, rewards, dones, values):
        """Salva un singolo step di transizione per tutti gli agenti."""
        self.states.append(states)
        self.global_states.append(global_state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)

    def get_tensors(self, device):
        """Converte i dati accumulati in Tensor PyTorch pronti per l'addestramento."""
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        global_states_t = torch.tensor(np.array(self.global_states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        log_probs_t = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        rewards_t = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        dones_t = torch.tensor(np.array(self.dones), dtype=torch.float32, device=device)
        values_t = torch.tensor(np.array(self.values), dtype=torch.float32, device=device)

        return states_t, global_states_t, actions_t, log_probs_t, rewards_t, dones_t, values_t