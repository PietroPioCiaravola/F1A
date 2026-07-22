import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Rete neurale locale dell'agente.
    Riceve lo stato locale (18 input) e restituisce la media e la deviazione standard
    per le azioni di guida (sterzo e acceleratore/freno).
    """
    def __init__(self, state_dim=18, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Bias iniziale sull'acceleratore (+1.5) per favorire l'avanzamento
        with torch.no_grad():
            self.mean.bias[1].fill_(1.5)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        clamped_log_std = torch.clamp(self.log_std, min=-1.6, max=0.5)
        std = torch.exp(clamped_log_std)
        return mean, std

class Critic(nn.Module):
    """
    Critic centralizzato (MAPPO).
    Riceve lo stato unito di tutti gli agenti (36 input) e restituisce 
    una stima del valore per ciascun agente (2 output).
    """
    def __init__(self, global_state_dim=36, num_agents=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_agents)
        )

    def forward(self, global_state):
        return self.net(global_state)