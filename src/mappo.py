import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MAPPO:
    def __init__(self, actor, critic, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, lmbda=0.95, eps_clip=0.2, k_epochs=1):
        self.actor = actor
        self.critic = critic
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def update(self, buffer, next_global_state, device):
        """Esegue l'aggiornamento delle reti neurali."""
        self.actor.train()
        self.critic.train()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. Recupera i tensor dal buffer
        states, global_states, actions, old_log_probs, rewards, dones, values = buffer.get_tensors(device)
        
        T = rewards.shape[0]
        if T == 0:
            return 0.0, 0.0
            
        num_agents = buffer.num_agents

        # 2. Bootstrap con l'ultimo stato globale tramite il Critic
        with torch.no_grad():
            next_global_state_tensor = torch.from_numpy(next_global_state).float().to(device).unsqueeze(0)
            next_value = self.critic(next_global_state_tensor).detach()

        # 3. Calcolo GAE e Returns blindato contro il trajectory bleeding
        values_extended = torch.cat([values, next_value], dim=0)
        advantages = torch.zeros(T, num_agents, device=device)
        gae = torch.zeros(num_agents, device=device)

        with torch.no_grad():
            # Converte dones in un tensore float sul device corretto una volta sola prima del ciclo
            dones_device = dones.float().to(device)

            for t in reversed(range(T)):
                # Flag non_terminal: 0.0 se l'episodio è finito (done=True), 1.0 altrimenti
                non_terminal = 1.0 - dones_device[t] # Operazione nativa tra tensori dello stesso device
                val_next = values_extended[t+1]

                # Delta di Bellman con protezione crash
                delta = rewards[t] + self.gamma * val_next * non_terminal - values_extended[t]
                gae = delta + self.gamma * self.lmbda * gae * non_terminal
                advantages[t] = gae

        returns = advantages + values

        # Normalizzazione dei vantaggi per singolo agente
        for a in range(num_agents):
            if advantages[:, a].std() > 0:
                advantages[:, a] = (advantages[:, a] - advantages[:, a].mean()) / (advantages[:, a].std() + 1e-8)

        # Spatelliamo le dimensioni per l'Actor
        states_flat = states.reshape(-1, buffer.state_dim)
        actions_flat = actions.reshape(-1, buffer.action_dim)
        old_log_probs_flat = old_log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)

        last_actor_loss, last_critic_loss = 0.0, 0.0

        # 4. Ciclo di ottimizzazione per k_epochs
        for _ in range(self.k_epochs):
            # --- Aggiornamento Actor ---
            mean, std = self.actor(states_flat)
            dist = Normal(mean, std)

            # Trasformazione log_prob per azioni continue limitate con atanh
            actions_clamped = torch.clamp(actions_flat, -0.999, 0.999)
            u = torch.atanh(actions_clamped)

            log_prob_u = dist.log_prob(u)
            log_prob = log_prob_u - torch.log(1 - actions_clamped.pow(2) + 1e-6)
            new_log_probs_flat = log_prob.sum(dim=1)

            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)

            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_flat

            entropy = dist.entropy().sum(dim=1).mean()
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # --- Aggiornamento Critic ---
            critic_values = self.critic(global_states)
            critic_loss = F.mse_loss(critic_values, returns.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()

        # Logging delle metriche
        print(f"📉 Critic loss: {last_critic_loss:.4f} | Actor loss: {last_actor_loss:.4f} | Mean reward: {rewards.mean().item():.4f}")

        # Salva i pesi delle reti
        torch.save(self.actor.state_dict(), "actor.pth")
        torch.save(self.critic.state_dict(), "critic.pth")
        print("💾 Models saved")

        # Resetta il buffer per raccogliere nuove transizioni
        buffer.reset()

        return last_actor_loss, last_critic_loss