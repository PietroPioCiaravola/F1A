import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Rete neurale locale dell'agente.
    Riceve lo stato locale (18 input) e restituisce la media e la deviazione standard
    per le azioni di guida (sterzo e acceleratore/freno).
    """
    def init(self, state_dim=18, action_dim=2):
        super().init()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mean = torch.tanh(self.mean(x))
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    """
    Critic centralizzato (MAPPO).
    Riceve lo stato unito di tutti gli agenti (36 input) e restituisce 
    una stima del valore per ciascun agente (2 output).
    """
    def init(self, global_state_dim=36, num_agents=2):
        super().init()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_agents)
        )

    def forward(self, global_state):
        return self.net(global_state)