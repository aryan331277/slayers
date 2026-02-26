"""
WorldSim India — MAPPO Training
=================================
Multi-Agent Proximal Policy Optimization (MAPPO) implemented from scratch
in pure PyTorch/NumPy — no RLlib dependency required.

Architecture:
  • Centralised critic   (sees all agents' observations concatenated)
  • Decentralised actors (each agent sees only its own 74-dim observation)
  • Shared actor weights (parameter sharing across all 10 state-agents)
  • GAE-Lambda advantage estimation
  • PPO clipped surrogate objective with entropy bonus
  • Separate Adam optimisers for actor and critic

Design references:
  Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative,
    Multi-Agent Games" (MAPPO paper) — arXiv:2103.01955
  Schulman et al. (2017) "Proximal Policy Optimization" — arXiv:1707.06347
  Schulman et al. (2015) "High-Dimensional Continuous Control Using
    Generalized Advantage Estimation" — arXiv:1506.02438

Usage (Kaggle):
  # install once
  !pip install pettingzoo gymnasium networkx scikit-learn torch -q

  # then run
  python worldsim_mappo.py
  # or import and call
  from worldsim_mappo import MAPPOTrainer
  trainer = MAPPOTrainer("worldsim_merged.csv")
  trainer.train(n_episodes=500)
"""

# ── Dependency check ──────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["torch", "pettingzoo", "gymnasium", "networkx"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import os, time, copy, warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

# Import our environment
# from worldsim_india import (
#     WorldSimIndiaEnv, AGENT_IDS, STATE_AGENTS,
#     N_ACTIONS, CLIMATE_STATES,
# )

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MAPPO] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MAPPOConfig:
    """All training hyperparameters in one place."""

    # ── Environment ────────────────────────────────────────────────────────
    csv_path:          str   = "worldsim_merged.csv"
    max_cycles:        int   = 150      # steps per episode
    noise_level:       float = 0.30     # observation noise (partial observability)
    reward_clip:       float = 5.0      # Clip rewards to [-clip, clip] for stability

    # ── Training schedule ──────────────────────────────────────────────────
    n_episodes:        int   = 600      # total training episodes
    n_epochs:          int   = 4        # PPO update epochs per episode (reduced from 8 for stability)
    minibatch_size:    int   = 128      # minibatch size for PPO update (increased from 64)
    eval_every:        int   = 50       # evaluate every N episodes (no exploration)
    save_every:        int   = 100      # save checkpoint every N episodes
    checkpoint_dir:    str   = "checkpoints"

    # ── Network architecture ───────────────────────────────────────────────
    obs_dim:           int   = 74       # must match WorldSimIndiaEnv.OBS_DIM
    n_agents:          int   = 10
    n_actions:         int   = N_ACTIONS   # 17
    n_targets:         int   = 10          # one per agent (target selection)
    actor_hidden:      List  = field(default_factory=lambda: [256, 256])
    critic_hidden:     List  = field(default_factory=lambda: [512, 512])

    # ── PPO ────────────────────────────────────────────────────────────────
    gamma:             float = 0.995     # discount factor (increased from 0.99 for longer horizon)
    gae_lambda:        float = 0.95     # GAE-λ
    clip_eps:          float = 0.15     # PPO clip ε (reduced from 0.20 for more conservative updates)
    entropy_coeff:     float = 0.02     # entropy bonus coefficient (slightly increased)
    value_coeff:       float = 1.0      # value loss coefficient (increased from 0.50)
    max_grad_norm:     float = 0.5      # gradient clipping (tightened from 10.0)
    target_kl:         float = 0.01     # Target KL divergence for early stopping

    # ── Optimiser ──────────────────────────────────────────────────────────
    lr_actor:          float = 5e-5     # Further reduced
    lr_critic:         float = 1e-4     # Further reduced
    lr_decay:          float = 0.999    # Slower decay

    # ── Exploration ────────────────────────────────────────────────────────
    # Linear decay of entropy coefficient:  end_entropy after warmup_episodes
    entropy_start:     float = 0.1      # Increased from 0.05
    entropy_end:       float = 0.01     # Increased from 0.005
    entropy_decay_eps: int   = 400

    # ── Normalisation ──────────────────────────────────────────────────────
    normalise_obs:     bool  = True     # running mean/std normalisation
    normalise_returns: bool  = True     # return normalisation
    normalise_rewards: bool  = True     # NEW: reward normalisation

    # ── Curriculum learning ────────────────────────────────────────────────
    # Start with cooperative reward (team avg), gradually shift to individual
    curriculum_start:  float = 0.70    # weight on team reward at episode 0
    curriculum_end:    float = 0.20    # weight on team reward at final episode


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NEURAL NETWORK ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

class RunningNorm(nn.Module):
    """
    Online running mean/std normalisation (Welford algorithm).
    Applied to observations before they enter the networks.
    Critical for stability in PPO when inputs have very different scales.
    """
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean",  torch.zeros(dim))
        self.register_buffer("var",   torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x: torch.Tensor):
        """Update statistics with a batch of observations (no grad required)."""
        with torch.no_grad():
            b_mean  = x.mean(0)
            b_var   = x.var(0, unbiased=False)
            b_count = torch.tensor(float(x.shape[0]), device=x.device)
            total   = self.count + b_count
            delta   = b_mean - self.mean
            new_mean = self.mean + delta * b_count / total
            m_a = self.var   * self.count
            m_b = b_var      * b_count
            m2  = m_a + m_b + delta ** 2 * self.count * b_count / total
            self.mean  = new_mean
            self.var   = m2 / total
            self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var + 1e-8)
        return torch.clamp((x - self.mean) / std, -self.clip, self.clip)


class ActorNetwork(nn.Module):
    """
    Decentralised actor: maps one agent's 74-dim observation →
        (action_type logits [17], target logits [10]).

    Parameter-shared across all 10 agents (MAPPO standard practice).
    The agent's one-hot ID is appended to the observation so the shared
    network can learn state-specific policies.

    Architecture: obs+id → LayerNorm → FC → ReLU → FC → ReLU → dual heads
    """
    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, n_targets: int,
                 hidden: List[int]):
        super().__init__()
        input_dim = obs_dim + n_agents  # obs + one-hot agent ID

        layers: List[nn.Module] = [nn.LayerNorm(input_dim)]
        in_d = input_dim
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        self.backbone = nn.Sequential(*layers)

        # Dual head: separate logits for (action_type, target_agent)
        self.action_head = nn.Linear(in_d, n_actions)
        self.target_head = nn.Linear(in_d, n_targets)

        # Orthogonal initialisation (recommended for PPO actors)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Output heads: smaller scale
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.orthogonal_(self.target_head.weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,          # (batch, obs_dim)
        agent_id: torch.Tensor,     # (batch,) — integer agent index
        action_mask: Optional[torch.Tensor] = None,   # (batch, n_actions)
        target_mask: Optional[torch.Tensor] = None,   # (batch, n_targets)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, target_logits)."""
        one_hot = F.one_hot(agent_id, num_classes=self.backbone[0].normalized_shape[0]
                            - obs.shape[-1]).float()
        x = torch.cat([obs, one_hot], dim=-1)
        feat = self.backbone(x)
        act_logits = self.action_head(feat)
        tgt_logits = self.target_head(feat)

        # Mask unavailable actions (e.g. do_nothing mask is never needed)
        if action_mask is not None:
            act_logits = act_logits.masked_fill(action_mask == 0, -1e9)
        if target_mask is not None:
            tgt_logits = tgt_logits.masked_fill(target_mask == 0, -1e9)

        return act_logits, tgt_logits

    def get_action(
        self,
        obs: torch.Tensor,
        agent_id: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or take greedy action.
        Returns: action_type, target_idx, log_prob, entropy
        """
        act_logits, tgt_logits = self.forward(obs, agent_id)
        act_dist = Categorical(logits=act_logits)
        tgt_dist = Categorical(logits=tgt_logits)

        if deterministic:
            act = act_logits.argmax(-1)
            tgt = tgt_logits.argmax(-1)
        else:
            act = act_dist.sample()
            tgt = tgt_dist.sample()

        # Joint log-probability: log P(action, target) = log P(action) + log P(target)
        log_prob = act_dist.log_prob(act) + tgt_dist.log_prob(tgt)
        entropy  = act_dist.entropy() + tgt_dist.entropy()
        return act, tgt, log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralised critic (CTDE — Centralised Training Decentralised Execution).
    Input: all agents' observations concatenated → scalar value estimate.
    This is the key MAPPO innovation: the critic has global state access during
    training, allowing much better advantage estimation than independent critics.

    Architecture: global_obs → LayerNorm → FC → ReLU → FC → ReLU → value
    """
    def __init__(self, obs_dim: int, n_agents: int, hidden: List[int]):
        super().__init__()
        global_dim = obs_dim * n_agents  # concatenate all obs

        layers: List[nn.Module] = [nn.LayerNorm(global_dim)]
        in_d = global_dim
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Value head
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        global_obs: (batch, obs_dim * n_agents) — all agents' obs concatenated.
        Returns:    (batch, 1) value estimates.
        """
        return self.net(global_obs)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transition:
    """Single agent-step transition stored in the rollout buffer."""
    obs:         np.ndarray   # (obs_dim,)
    global_obs:  np.ndarray   # (obs_dim * n_agents,)
    agent_id:    int
    action_type: int
    target_idx:  int
    log_prob:    float
    reward:      float
    done:        bool
    value:       float


class RolloutBuffer:
    """
    Stores one complete episode worth of transitions per agent.
    Computes GAE-λ advantage estimates at episode end.
    """

    def __init__(self, config: MAPPOConfig):
        self.cfg = config
        self.clear()

    def clear(self):
        self.transitions: List[Transition] = []

    def push(self, t: Transition):
        self.transitions.append(t)

    def compute_advantages_and_returns(
        self, last_values: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GAE-λ advantage estimation (Schulman et al. 2015).
        Processes each agent's timeline independently, then flattens.

        Returns advantages (N,) and returns (N,) for all stored transitions.
        """
        cfg = self.cfg
        n   = len(self.transitions)
        if n == 0:
            return np.array([]), np.array([])

        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)

        # Group by agent for per-agent GAE
        agent_indices: Dict[int, List[int]] = defaultdict(list)
        for i, t in enumerate(self.transitions):
            agent_indices[t.agent_id].append(i)

        for aid_idx, idxs in agent_indices.items():
            aid = AGENT_IDS[aid_idx]
            vals = [self.transitions[i].value for i in idxs]
            rwds = [self.transitions[i].reward for i in idxs]
            done = [self.transitions[i].done   for i in idxs]

            # Bootstrap from last value if episode didn't end in collapse
            last_v = last_values.get(aid, 0.0) if not done[-1] else 0.0

            gae = 0.0
            for j in reversed(range(len(idxs))):
                next_v = vals[j + 1] if j + 1 < len(idxs) else last_v
                mask   = 0.0 if done[j] else 1.0
                delta  = rwds[j] + cfg.gamma * next_v * mask - vals[j]
                gae    = delta + cfg.gamma * cfg.gae_lambda * mask * gae
                advantages[idxs[j]] = gae
                returns[idxs[j]]    = gae + vals[j]

        return advantages, returns

    def get_tensors(
        self, advantages: np.ndarray, returns: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Collate buffer into training tensors."""
        obs         = np.stack([t.obs        for t in self.transitions])
        global_obs  = np.stack([t.global_obs for t in self.transitions])
        agent_ids   = np.array([t.agent_id   for t in self.transitions])
        act_types   = np.array([t.action_type for t in self.transitions])
        tgt_idxs    = np.array([t.target_idx  for t in self.transitions])
        log_probs   = np.array([t.log_prob    for t in self.transitions])
        values      = np.array([t.value       for t in self.transitions])   # ← ADD THIS

        return {
            "obs":        torch.FloatTensor(obs).to(DEVICE),
            "global_obs": torch.FloatTensor(global_obs).to(DEVICE),
            "agent_ids":  torch.LongTensor(agent_ids).to(DEVICE),
            "act_types":  torch.LongTensor(act_types).to(DEVICE),
            "tgt_idxs":   torch.LongTensor(tgt_idxs).to(DEVICE),
            "log_probs":  torch.FloatTensor(log_probs).to(DEVICE),
            "advantages": torch.FloatTensor(advantages).to(DEVICE),
            "returns":    torch.FloatTensor(returns).to(DEVICE),
            "values":     torch.FloatTensor(values).to(DEVICE),          # ← NEW
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MAPPO TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class MAPPOTrainer:
    """
    Full MAPPO training loop for WorldSim India.

    Training loop per episode:
      1.  Reset environment
      2.  Roll out episode, collecting (obs, action, reward, value) per agent step
      3.  Compute GAE-λ advantages and returns
      4.  Run n_epochs PPO update passes over minibatches
      5.  Log metrics, decay LR, decay entropy coefficient
      6.  Periodically evaluate and save checkpoint

    The centralised critic sees ALL agents' observations concatenated
    (global state). Only the actor is deployed at inference time.
    """

    def __init__(self, csv_path: str, config: Optional[MAPPOConfig] = None):
        self.cfg = config or MAPPOConfig(csv_path=csv_path)
        self.cfg.csv_path = csv_path

        # ── Environment ──────────────────────────────────────────────────────
        self.env = WorldSimIndiaEnv(
            csv_path=self.cfg.csv_path,
            max_cycles=self.cfg.max_cycles,
            noise_level=self.cfg.noise_level,
        )

        # ── Networks ─────────────────────────────────────────────────────────
        self.actor = ActorNetwork(
            obs_dim   = self.cfg.obs_dim,
            n_agents  = self.cfg.n_agents,
            n_actions = self.cfg.n_actions,
            n_targets = self.cfg.n_targets,
            hidden    = self.cfg.actor_hidden,
        ).to(DEVICE)

        self.critic = CriticNetwork(
            obs_dim   = self.cfg.obs_dim,
            n_agents  = self.cfg.n_agents,
            hidden    = self.cfg.critic_hidden,
        ).to(DEVICE)

        # ── Observation normaliser (shared across all agents) ─────────────────
        self.obs_norm = RunningNorm(self.cfg.obs_dim).to(DEVICE)

        # ── Reward normaliser ─────────────────────────────────────────────────
        if self.cfg.normalise_rewards:
            self.reward_norm = RunningNorm(1).to(DEVICE)  # scalar rewards

        # ── Optimisers ────────────────────────────────────────────────────────
        self.actor_opt  = Adam(self.actor.parameters(),  lr=self.cfg.lr_actor)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        # ── LR schedulers ─────────────────────────────────────────────────────
        self.actor_sched  = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_opt,  gamma=self.cfg.lr_decay)
        self.critic_sched = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_opt, gamma=self.cfg.lr_decay)

        # ── Buffer ────────────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(self.cfg)

        # ── Metrics ───────────────────────────────────────────────────────────
        self.episode_returns:    List[Dict[str, float]] = []
        self.episode_lengths:    List[int]              = []
        self.episode_survivors:  List[int]              = []
        self.ppo_losses:         List[Dict[str, float]] = []
        self.eval_returns:       List[float]            = []
        self._best_eval_return:  float                  = -np.inf

        # ── Misc ──────────────────────────────────────────────────────────────
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self._episode = 0

        n_actor  = sum(p.numel() for p in self.actor.parameters()  if p.requires_grad)
        n_critic = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        print(f"\n[MAPPO] Actor params:  {n_actor:,}")
        print(f"[MAPPO] Critic params: {n_critic:,}")
        print(f"[MAPPO] Config: episodes={self.cfg.n_episodes}  "
              f"cycles/ep={self.cfg.max_cycles}  n_epochs={self.cfg.n_epochs}")

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _entropy_coeff(self) -> float:
        """Linearly decay entropy coefficient from start to end."""
        frac = min(1.0, self._episode / max(1, self.cfg.entropy_decay_eps))
        return self.cfg.entropy_start + frac * (
            self.cfg.entropy_end - self.cfg.entropy_start)

    def _team_weight(self) -> float:
        """Curriculum: linearly decay team reward weighting."""
        frac = min(1.0, self._episode / max(1, self.cfg.n_episodes))
        return self.cfg.curriculum_start + frac * (
            self.cfg.curriculum_end - self.cfg.curriculum_start)

    def _get_global_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all agents' observations into global state vector."""
        return np.concatenate(
            [obs_dict.get(aid, np.zeros(self.cfg.obs_dim, dtype=np.float32))
             for aid in AGENT_IDS]
        )

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        if self.cfg.normalise_obs:
            t = self.obs_norm(t)
        return t

    @torch.no_grad()
    def _get_value(self, global_obs_np: np.ndarray) -> float:
        t = torch.FloatTensor(global_obs_np).unsqueeze(0).to(DEVICE)
        return float(self.critic(t).squeeze())

    # ─────────────────────────────────────────────────────────────────────────
    # COLLECT ROLLOUT
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect_rollout(self, deterministic: bool = False) -> Dict[str, float]:
        """
        Run one full episode, storing transitions in the buffer.
        Returns per-agent episode returns for logging.
        """
        self.buffer.clear()
        obs_dict, _ = self.env.reset(seed=self._episode if not deterministic else 9999)

        episode_returns: Dict[str, float] = defaultdict(float)
        current_obs = dict(obs_dict)  # mutable copy

        # Update obs normaliser with initial observations
        if self.cfg.normalise_obs and not deterministic:
            obs_batch = torch.FloatTensor(
                np.stack(list(obs_dict.values()))).to(DEVICE)
            self.obs_norm.update(obs_batch)

        for step in range(self.cfg.max_cycles):
            if not self.env.agents:
                break

            # Active agent this AEC step
            agent = self.env.agent_selection
            if self.env.terminations.get(agent, False) or \
               self.env.truncations.get(agent, False):
                try:
                    self.env.step(np.array([16, 0]))  # do_nothing for dead agent
                except Exception:
                    pass
                continue

            aid_idx = AGENT_IDS.index(agent)
            obs_np  = current_obs.get(agent, np.zeros(self.cfg.obs_dim, dtype=np.float32))

            # Build global obs for critic
            global_obs_np = self._get_global_obs(current_obs)

            # Actor: sample action
            obs_t   = self._obs_to_tensor(obs_np)
            aid_t   = torch.LongTensor([aid_idx]).to(DEVICE)
            act, tgt, log_prob, _ = self.actor.get_action(
                obs_t, aid_t, deterministic=deterministic)

            action_type = int(act.item())
            target_idx  = int(tgt.item())
            log_prob_f  = float(log_prob.item())

            # Critic: estimate value
            value = self._get_value(global_obs_np)

            # Step environment
            action = np.array([action_type, target_idx])
            self.env.step(action)

            reward = float(self.env.rewards.get(agent, 0.0))
            done   = (self.env.terminations.get(agent, False) or
                      self.env.truncations.get(agent, False))

            # NEW: Clip reward for stability
            reward = np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

            # NEW: Normalize reward if enabled
            if self.cfg.normalise_rewards and not deterministic:
                reward_t = torch.tensor([[reward]]).to(DEVICE)
                self.reward_norm.update(reward_t)
                reward = float(self.reward_norm(reward_t).item())

            # Curriculum: blend individual + team reward
            team_w = self._team_weight()
            if not deterministic and len(self.env.agents) > 0:
                team_reward = float(np.mean(
                    [self.env.rewards.get(a, 0.0) for a in self.env.agents]))
                team_reward = np.clip(team_reward, -self.cfg.reward_clip, self.cfg.reward_clip)
                reward = (1 - team_w) * reward + team_w * team_reward

            episode_returns[agent] += reward

            # Store transition
            self.buffer.push(Transition(
                obs        = obs_np,
                global_obs = global_obs_np,
                agent_id   = aid_idx,
                action_type= action_type,
                target_idx = target_idx,
                log_prob   = log_prob_f,
                reward     = reward,
                done       = done,
                value      = value,
            ))

            # Update current obs
            if not done:
                new_obs = self.env._observe(agent)
                current_obs[agent] = new_obs
                if self.cfg.normalise_obs and not deterministic:
                    t = torch.FloatTensor(new_obs).unsqueeze(0).to(DEVICE)
                    self.obs_norm.update(t)

        return dict(episode_returns)

    # ─────────────────────────────────────────────────────────────────────────
    # PPO UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def ppo_update(self) -> Dict[str, float]:
        """
        Run n_epochs PPO updates over minibatches of the stored rollout.

        PPO clipped objective (Schulman et al. 2017):
          L_CLIP = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]
        where r_t = π(a|s) / π_old(a|s) is the probability ratio.

        Combined loss:
          L = -L_CLIP + v_coeff × L_VALUE - entropy_coeff × H[π]

        Improvements: 
        - Value clipping for stability
        - Early stopping if KL divergence exceeds target
        - Approx KL monitoring
        """
        # Bootstrap values for agents still alive at episode end
        last_values = {}
        if self.cfg.normalise_obs:
            for aid in AGENT_IDS:
                if aid in self.env.agents:
                    global_obs = self._get_global_obs(
                        {a: self.env._observe(a) for a in self.env.agents})
                    last_values[aid] = self._get_value(global_obs)
                else:
                    last_values[aid] = 0.0
        else:
            last_values = {aid: 0.0 for aid in AGENT_IDS}

        advantages, returns = self.buffer.compute_advantages_and_returns(last_values)

        if len(advantages) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0, "approx_kl": 0}

        # Normalise advantages (reduces variance)
        if self.cfg.normalise_returns:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        batch = self.buffer.get_tensors(advantages, returns)
        n     = len(advantages)
        ent_c = self._entropy_coeff()

        metrics: Dict[str, List[float]] = defaultdict(list)

        for epoch in range(self.cfg.n_epochs):
            # Shuffle indices for minibatch sampling
            perm = torch.randperm(n, device=DEVICE)
            approx_kl = 0.0

            for start in range(0, n, self.cfg.minibatch_size):
                idx = perm[start: start + self.cfg.minibatch_size]
                if len(idx) < 2:
                    continue

                obs       = batch["obs"][idx]
                glob_obs  = batch["global_obs"][idx]
                agent_ids = batch["agent_ids"][idx]
                act_types = batch["act_types"][idx]
                tgt_idxs  = batch["tgt_idxs"][idx]
                old_lp    = batch["log_probs"][idx]
                adv       = batch["advantages"][idx]
                ret       = batch["returns"][idx]

                # ── Actor loss ────────────────────────────────────────────────
                act_logits, tgt_logits = self.actor(obs, agent_ids)
                act_dist = Categorical(logits=act_logits)
                tgt_dist = Categorical(logits=tgt_logits)

                new_lp = act_dist.log_prob(act_types) + \
                         tgt_dist.log_prob(tgt_idxs)
                entropy = act_dist.entropy() + tgt_dist.entropy()

                # Probability ratio
                ratio = torch.exp(new_lp - old_lp)

                # Clipped surrogate
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps,
                                    1 + self.cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Critic loss ───────────────────────────────────────────────
                values = self.critic(glob_obs).squeeze(-1)
                old_values = batch["values"][idx]
                values_clipped = old_values + torch.clamp(
                    values - old_values, -self.cfg.clip_eps, self.cfg.clip_eps
                )

                # More stable version — penalize the worse deviation
                value_loss = torch.mean(
                    torch.max(
                        (values - ret) ** 2,
                        (values_clipped - ret) ** 2
                    )
                )

                # ── Combined loss ─────────────────────────────────────────────
                loss = (policy_loss
                        + self.cfg.value_coeff  * value_loss
                        - ent_c                 * entropy.mean())

                # ── Approx KL for early stopping ──────────────────────────────
                kl = (old_lp - new_lp).mean().item()
                approx_kl += kl

                # ── Backprop ──────────────────────────────────────────────────
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) +
                    list(self.critic.parameters()),
                    self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                metrics["policy_loss"].append(float(policy_loss))
                metrics["value_loss"].append(float(value_loss))
                metrics["entropy"].append(float(entropy.mean()))
                metrics["total_loss"].append(float(loss))
                metrics["ratio_mean"].append(float(ratio.mean()))
                metrics["clip_frac"].append(
                    float((torch.abs(ratio - 1) > self.cfg.clip_eps).float().mean()))

            # Early stopping if KL too high
            approx_kl /= (n / self.cfg.minibatch_size)
            metrics["approx_kl"].append(approx_kl)
            if approx_kl > self.cfg.target_kl:
                break

        return {k: float(np.mean(v)) for k, v in metrics.items() if v}

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATE (no exploration noise, deterministic policy)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, n_eval: int = 3) -> Dict[str, float]:
        """
        Run n_eval deterministic episodes and return aggregate metrics.
        Captures the five emergent behaviours described in the WorldSim blueprint.
        """
        self.actor.eval()
        self.critic.eval()

        agg_returns:   List[float] = []
        agg_survivors: List[int]   = []
        agg_alliances: List[float] = []
        agg_trades:    List[float] = []
        agg_conflicts: List[float] = []

        for _ in range(n_eval):
            ep_returns = self.collect_rollout(deterministic=True)
            agg_returns.append(float(np.mean(list(ep_returns.values()))))
            agg_survivors.append(len(self.env.agents))
            agg_alliances.append(
                float(np.mean([len(self.env._alliances[a])
                               for a in AGENT_IDS])))
            agg_trades.append(float(len(self.env._trade_agreements)))
            agg_conflicts.append(
                float(self.env._conflict_matrix.mean()))

        self.actor.train()
        self.critic.train()

        return {
            "eval_return":     float(np.mean(agg_returns)),
            "eval_survivors":  float(np.mean(agg_survivors)),
            "eval_alliances":  float(np.mean(agg_alliances)),
            "eval_trades":     float(np.mean(agg_trades)),
            "eval_conflict":   float(np.mean(agg_conflicts)),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CHECKPOINT
    # ─────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str = ""):
        fname = os.path.join(
            self.cfg.checkpoint_dir,
            f"worldsim_india_ep{self._episode}{('_' + tag) if tag else ''}.pt")
        torch.save({
            "episode":      self._episode,
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "actor_opt":    self.actor_opt.state_dict(),
            "critic_opt":   self.critic_opt.state_dict(),
            "obs_norm":     self.obs_norm.state_dict(),
            "reward_norm":  self.reward_norm.state_dict() if self.cfg.normalise_rewards else None,
            "config":       self.cfg,
            "best_eval":    self._best_eval_return,
        }, fname)
        print(f"  [ckpt] Saved → {fname}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.obs_norm.load_state_dict(ckpt["obs_norm"])
        if self.cfg.normalise_rewards and "reward_norm" in ckpt:
            self.reward_norm.load_state_dict(ckpt["reward_norm"])
        self._episode = ckpt["episode"]
        self._best_eval_return = ckpt.get("best_eval", -np.inf)
        print(f"  [ckpt] Loaded from {path} (episode {self._episode})")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, n_episodes: Optional[int] = None) -> pd.DataFrame:
        """
        Main training loop.  Returns a DataFrame of training metrics per episode.

        Per episode:
          1. collect_rollout  → fills buffer, returns episode_returns dict
          2. ppo_update       → runs n_epochs of PPO over the buffer
          3. log metrics
          4. (optional) evaluate → deterministic policy assessment
          5. (optional) save checkpoint
        """
        n_eps = n_episodes or self.cfg.n_episodes
        log_rows: List[dict] = []

        # Rolling windows for smoothed logging
        ret_window   = deque(maxlen=20)
        surv_window  = deque(maxlen=20)

        print(f"\n{'='*70}")
        print(f"  WorldSim India MAPPO Training")
        print(f"  Episodes: {n_eps}  |  Cycles/ep: {self.cfg.max_cycles}")
        print(f"  Actor arch: {self.cfg.actor_hidden}  "
              f"Critic arch: {self.cfg.critic_hidden}")
        print(f"{'='*70}")
        print(f"  {'Ep':>5} {'MeanRet':>9} {'SmoothR':>9} {'Surv':>5} "
              f"{'PLoss':>8} {'VLoss':>8} {'Ent':>7} "
              f"{'Clip%':>7} {'KL':>7} {'LR_a':>8} {'Time':>6}")
        print(f"  {'─'*5} {'─'*9} {'─'*9} {'─'*5} "
              f"{'─'*8} {'─'*8} {'─'*7} "
              f"{'─'*7} {'─'*7} {'─'*8} {'─'*6}")

        t0_total = time.time()

        for ep in range(n_eps):
            self._episode = ep
            t0_ep = time.time()

            # 1. Collect rollout
            ep_returns = self.collect_rollout(deterministic=False)
            n_alive    = len(self.env.agents)
            mean_ret   = float(np.mean(list(ep_returns.values())))

            # 2. PPO update (skip if buffer empty)
            if len(self.buffer.transitions) > 0:
                loss_dict = self.ppo_update()
            else:
                loss_dict = {"policy_loss": 0, "value_loss": 0,
                             "entropy": 0, "clip_frac": 0, "approx_kl": 0}

            # 3. LR decay
            self.actor_sched.step()
            self.critic_sched.step()

            # 4. Rolling window logging
            ret_window.append(mean_ret)
            surv_window.append(n_alive)
            smooth_ret = float(np.mean(ret_window))

            ep_data = {
                "episode":       ep,
                "mean_return":   mean_ret,
                "smooth_return": smooth_ret,
                "n_survivors":   n_alive,
                "policy_loss":   loss_dict.get("policy_loss", 0),
                "value_loss":    loss_dict.get("value_loss", 0),
                "entropy":       loss_dict.get("entropy", 0),
                "clip_frac":     loss_dict.get("clip_frac", 0),
                "approx_kl":     loss_dict.get("approx_kl", 0),
                "lr_actor":      self.actor_opt.param_groups[0]["lr"],
                "entropy_coeff": self._entropy_coeff(),
                "team_weight":   self._team_weight(),
                "ep_time_s":     time.time() - t0_ep,
            }
            ep_data.update({f"ret_{aid}": ep_returns.get(aid, 0)
                            for aid in AGENT_IDS})
            log_rows.append(ep_data)

            # Print progress
            if ep % 10 == 0 or ep == n_eps - 1:
                print(
                    f"  {ep:>5} {mean_ret:>9.2f} {smooth_ret:>9.2f} "
                    f"{n_alive:>5d} "
                    f"{loss_dict.get('policy_loss',0):>8.4f} "
                    f"{loss_dict.get('value_loss',0):>8.4f} "
                    f"{loss_dict.get('entropy',0):>7.4f} "
                    f"{loss_dict.get('clip_frac',0)*100:>6.1f}% "
                    f"{loss_dict.get('approx_kl',0):>7.4f} "
                    f"{self.actor_opt.param_groups[0]['lr']:>8.2e} "
                    f"{time.time()-t0_ep:>5.1f}s"
                )

            # 5. Evaluation
            if (ep + 1) % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate(n_eval=3)
                self.eval_returns.append(eval_metrics["eval_return"])
                ep_data.update(eval_metrics)
                new_best = eval_metrics["eval_return"] > self._best_eval_return
                if new_best:
                    self._best_eval_return = eval_metrics["eval_return"]
                    self.save_checkpoint("best")
                print(f"\n  [EVAL ep={ep+1}] "
                      f"return={eval_metrics['eval_return']:.2f}  "
                      f"survivors={eval_metrics['eval_survivors']:.1f}  "
                      f"alliances={eval_metrics['eval_alliances']:.2f}  "
                      f"trades={eval_metrics['eval_trades']:.1f}  "
                      f"conflict={eval_metrics['eval_conflict']:.3f}"
                      + ("  ← NEW BEST" if new_best else "") + "\n")

            # 6. Periodic checkpoint
            if (ep + 1) % self.cfg.save_every == 0:
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint("final")

        total_time = time.time() - t0_total
        print(f"\n{'='*70}")
        print(f"  Training complete — {n_eps} episodes in "
              f"{total_time/60:.1f} min")
        print(f"  Best eval return: {self._best_eval_return:.2f}")
        print(f"{'='*70}")

        return pd.DataFrame(log_rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(log_df: pd.DataFrame, save_path: str = "training_curves.png"):
    """
    Plot 8 training diagnostic panels from the training log DataFrame.
    Call after trainer.train() returns log_df.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.patch.set_facecolor("#0a0a0a")
    fig.suptitle("WorldSim India — MAPPO Training Diagnostics",
                 color="white", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    COLORS = {
        "RJ":"#e74c3c","MH":"#3498db","UP":"#f39c12","KL":"#2ecc71",
        "GJ":"#9b59b6","WB":"#1abc9c","PB":"#e67e22","BR":"#e91e63",
        "KA":"#34495e","TN":"#16a085",
    }

    def dark(ax, title=""):
        ax.set_facecolor("#111111")
        ax.spines[:].set_color("#333333")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        if title:
            ax.set_title(title, color="white", fontsize=10)
        ax.grid(True, alpha=0.12, color="white")

    eps = log_df["episode"].values

    # 1: Smoothed return
    ax = axes[0]; dark(ax, "Episode Return (smoothed 20-ep window)")
    ax.plot(eps, log_df["mean_return"],   color="#555555", lw=0.8, alpha=0.5)
    ax.plot(eps, log_df["smooth_return"], color="#2ecc71", lw=2.0, label="Smooth return")
    ax.axhline(0, color="#ff6b6b", lw=0.8, ls=":")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.legend(facecolor="#111111", labelcolor="white", fontsize=8)

    # 2: Per-agent returns
    ax = axes[1]; dark(ax, "Per-Agent Episode Return")
    for aid in AGENT_IDS:
        col = f"ret_{aid}"
        if col in log_df.columns:
            smoothed = pd.Series(log_df[col]).rolling(20, min_periods=1).mean()
            ax.plot(eps, smoothed, color=COLORS[aid], lw=1.2,
                    label=STATE_AGENTS[aid][:10], alpha=0.85)
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.legend(fontsize=6, facecolor="#111111", labelcolor="white",
              bbox_to_anchor=(1.01,1), loc="upper left", ncol=1)

    # 3: Survivors
    ax = axes[2]; dark(ax, "Agents Alive at Episode End")
    ax.fill_between(eps, log_df["n_survivors"], alpha=0.3, color="#3498db")
    ax.plot(eps, log_df["n_survivors"], color="#3498db", lw=1.5)
    ax.plot(eps,
            pd.Series(log_df["n_survivors"]).rolling(20,min_periods=1).mean(),
            color="#f39c12", lw=2.0, label="Smoothed")
    ax.set_ylim(0, 11)
    ax.set_xlabel("Episode"); ax.set_ylabel("# Agents Alive")
    ax.legend(facecolor="#111111", labelcolor="white", fontsize=8)

    # 4: Policy loss
    ax = axes[3]; dark(ax, "Policy Loss")
    ax.plot(eps, log_df["policy_loss"], color="#e74c3c", lw=1.0, alpha=0.7)
    ax.plot(eps, pd.Series(log_df["policy_loss"]).rolling(20,min_periods=1).mean(),
            color="#ff9f43", lw=2.0)
    ax.set_xlabel("Episode"); ax.set_ylabel("Loss")

    # 5: Value loss
    ax = axes[4]; dark(ax, "Value (Critic) Loss")
    ax.plot(eps, log_df["value_loss"], color="#9b59b6", lw=1.0, alpha=0.7)
    ax.plot(eps, pd.Series(log_df["value_loss"]).rolling(20,min_periods=1).mean(),
            color="#d980fa", lw=2.0)
    ax.set_xlabel("Episode"); ax.set_ylabel("MSE Loss")

    # 6: Entropy & clip fraction
    ax = axes[5]; dark(ax, "Entropy & Clip Fraction")
    ax2 = ax.twinx()
    ax.plot(eps,  log_df["entropy"],   color="#1abc9c", lw=1.5, label="Entropy")
    ax2.plot(eps, log_df["clip_frac"] * 100, color="#e67e22", lw=1.0,
             ls="--", alpha=0.7, label="Clip %")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy",    color="#1abc9c")
    ax2.set_ylabel("Clip %",   color="#e67e22")
    ax.tick_params(axis="y", colors="#1abc9c")
    ax2.tick_params(axis="y", colors="#e67e22")
    ax2.spines[:].set_color("#333333")
    ax2.set_facecolor("#111111")

    # 7: LR decay
    ax = axes[6]; dark(ax, "Learning Rate Decay")
    ax.semilogy(eps, log_df["lr_actor"], color="#2980b9", lw=1.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Actor LR (log)")

    # 8: Curriculum / entropy coefficient
    ax = axes[7]; dark(ax, "Curriculum & Entropy Coefficient")
    ax.plot(eps, log_df["team_weight"],    color="#f39c12", lw=2.0,
            label="Team reward weight")
    ax.plot(eps, log_df["entropy_coeff"],  color="#2ecc71", lw=2.0,
            label="Entropy coeff")
    ax.set_xlabel("Episode"); ax.set_ylabel("Coefficient")
    ax.legend(facecolor="#111111", labelcolor="white", fontsize=8)

    # 9: Eval returns (sparse)
    ax = axes[8]; dark(ax, "Deterministic Eval Returns")
    eval_rows = log_df.dropna(subset=["eval_return"]) if "eval_return" in log_df.columns else pd.DataFrame()
    if len(eval_rows) > 0:
        ax.plot(eval_rows["episode"], eval_rows["eval_return"],
                "o-", color="#e74c3c", lw=2.0, ms=5, label="Eval return")
        ax.plot(eval_rows["episode"], eval_rows["eval_survivors"],
                "s--", color="#3498db", lw=1.5, ms=4, label="Eval survivors")
        ax.legend(facecolor="#111111", labelcolor="white", fontsize=8)
    ax.set_xlabel("Episode")

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    print(f"[MAPPO] Training curves saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STRATEGY ANALYSIS
# WorldSim Blueprint Feature 6: "Strategy Heatmap — the moment an agent
# figured out its only viable path"
# ══════════════════════════════════════════════════════════════════════════════

def analyse_emergent_strategies(
    trainer: MAPPOTrainer,
    n_eval_episodes: int = 20,
    save_path: str = "strategy_analysis.png",
):
    """
    Run n_eval_episodes of deterministic policy and extract:
      1. Action frequency heatmap per agent (strategy heatmap)
      2. Alliance formation patterns
      3. Defection history correlation with resource stress
      4. Survival rate by initial resource profile
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n[Analysis] Running {n_eval_episodes} eval episodes...")
    trainer.actor.eval()

    # Aggregate action counts per (agent, action_type)
    action_counts  = np.zeros((len(AGENT_IDS), N_ACTIONS), dtype=float)
    alliance_counts= np.zeros((len(AGENT_IDS), len(AGENT_IDS)), dtype=float)
    survival_record= defaultdict(list)
    defection_by_aid= defaultdict(int)
    trade_by_pair  = defaultdict(int)

    for trial in range(n_eval_episodes):
        obs_dict, _ = trainer.env.reset(seed=1000 + trial)
        current_obs = dict(obs_dict)

        for step in range(trainer.cfg.max_cycles):
            if not trainer.env.agents:
                break
            agent = trainer.env.agent_selection
            if trainer.env.terminations.get(agent, False) or \
               trainer.env.truncations.get(agent, False):
                try: trainer.env.step(np.array([16, 0]))
                except: pass
                continue

            aid_idx = AGENT_IDS.index(agent)
            obs_np  = current_obs.get(agent, np.zeros(trainer.cfg.obs_dim, dtype=np.float32))

            with torch.no_grad():
                obs_t = trainer._obs_to_tensor(obs_np)
                aid_t = torch.LongTensor([aid_idx]).to(DEVICE)
                act, tgt, _, _ = trainer.actor.get_action(obs_t, aid_t, deterministic=True)

            action_type = int(act.item())
            target_idx  = int(tgt.item())
            action_counts[aid_idx, action_type] += 1

            trainer.env.step(np.array([action_type, target_idx]))
            current_obs[agent] = trainer.env._observe(agent)

        # Record end-of-episode state
        for aid in AGENT_IDS:
            survived = aid in trainer.env.agents
            survival_record[aid].append(int(survived))
            n_allies = len(trainer.env._alliances.get(aid, set()))
            for ally in trainer.env._alliances.get(aid, set()):
                ai = AGENT_IDS.index(aid)
                aj = AGENT_IDS.index(ally) if ally in AGENT_IDS else -1
                if aj >= 0:
                    alliance_counts[ai, aj] += 1

        for (a, b) in trainer.env._trade_agreements:
            trade_by_pair[(a, b)] += 1
        for aid in AGENT_IDS:
            defection_by_aid[aid] += trainer.env._defection_count.get(aid, 0)

    trainer.actor.train()

    # ── Normalise ────────────────────────────────────────────────────────────
    row_sums = action_counts.sum(axis=1, keepdims=True)
    action_freq = action_counts / np.maximum(row_sums, 1)
    alliance_freq = alliance_counts / max(n_eval_episodes, 1)
    survival_rate = {aid: float(np.mean(v)) for aid, v in survival_record.items()}
    defection_rate = {aid: defection_by_aid[aid] / max(n_eval_episodes, 1)
                      for aid in AGENT_IDS}

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor("#0a0a0a")
    fig.suptitle("WorldSim India — Emergent Strategy Analysis",
                 color="white", fontsize=16, fontweight="bold")

    def dark(ax, title=""):
        ax.set_facecolor("#111111")
        ax.spines[:].set_color("#333333")
        ax.tick_params(colors="white")
        if title: ax.set_title(title, color="white", fontsize=11)

    action_labels = [ACTION_TYPES[i] for i in range(N_ACTIONS)]
    state_labels  = [STATE_AGENTS[a][:10] for a in AGENT_IDS]

    # 1: Strategy heatmap
    ax = axes[0, 0]; dark(ax, "Strategy Heatmap — Action Frequency per Agent")
    sns.heatmap(
        action_freq, ax=ax,
        xticklabels=action_labels, yticklabels=state_labels,
        cmap="YlOrRd", vmin=0, vmax=action_freq.max(),
        linewidths=0.3, linecolor="#222222",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7, colors="white")
    ax.tick_params(axis="y", labelsize=8, colors="white")

    # 2: Alliance network heatmap
    ax = axes[0, 1]; dark(ax, "Alliance Formation Frequency")
    mask = np.eye(len(AGENT_IDS), dtype=bool)
    sns.heatmap(
        alliance_freq, ax=ax,
        xticklabels=AGENT_IDS, yticklabels=AGENT_IDS,
        cmap="Blues", vmin=0,
        linewidths=0.5, linecolor="#333333",
        mask=mask, annot=True, fmt=".1f", annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(colors="white", labelsize=8)

    # 3: Survival rate + defection
    ax = axes[1, 0]; dark(ax, "Survival Rate & Defection Rate by State")
    x = np.arange(len(AGENT_IDS))
    surv_vals = [survival_rate[a] for a in AGENT_IDS]
    def_vals  = [defection_rate[a] for a in AGENT_IDS]
    bars = ax.bar(x - 0.2, surv_vals, 0.35, color="#2ecc71", alpha=0.85,
                  label="Survival rate")
    ax2  = ax.twinx()
    ax2.bar(x + 0.2, def_vals, 0.35, color="#e74c3c", alpha=0.70,
            label="Defection rate")
    ax.set_xticks(x); ax.set_xticklabels(AGENT_IDS, color="white", fontsize=9)
    ax.set_ylabel("Survival rate", color="#2ecc71")
    ax2.set_ylabel("Defections / episode", color="#e74c3c")
    ax.tick_params(axis="y", colors="#2ecc71")
    ax2.tick_params(axis="y", colors="#e74c3c")
    ax2.spines[:].set_color("#333333")
    ax2.set_facecolor("#111111")
    ax.set_ylim(0, 1.15)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2,
              facecolor="#111111", labelcolor="white", fontsize=8,
              loc="upper right")

    # 4: Initial resource profile vs survival
    ax = axes[1, 1]; dark(ax, "Initial Resource Profile vs Survival Rate")
    init   = trainer.env.data_loader.region_init
    labels = {"water_stock": "Water", "food_stock": "Food",
              "energy_stock": "Energy", "economic_power": "Economy"}
    colors_s = ["#3498db","#2ecc71","#f39c12","#e74c3c"]
    xs = np.arange(len(AGENT_IDS))
    for i, (key, lbl) in enumerate(labels.items()):
        vals = [init[a][key] for a in AGENT_IDS]
        ax.plot(xs, vals, "o-", color=colors_s[i], lw=1.5,
                label=lbl, alpha=0.85)
    ax.plot(xs, surv_vals, "s--", color="white", lw=2.0,
            ms=6, label="Survival rate")
    ax.set_xticks(xs)
    ax.set_xticklabels([STATE_AGENTS[a][:6] for a in AGENT_IDS],
                       color="white", fontsize=8)
    ax.set_ylabel("Score (0–1)", color="white")
    ax.legend(facecolor="#111111", labelcolor="white", fontsize=8)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    print(f"[Analysis] Strategy analysis saved → {save_path}")
    plt.show()

    # Print summary table
    print(f"\n{'─'*60}")
    print(f"  {'State':<22} {'Survival':>9} {'Defects/ep':>11} "
          f"{'Top Action':<20}")
    print(f"  {'─'*22} {'─'*9} {'─'*11} {'─'*20}")
    for aid in AGENT_IDS:
        top_act_idx = int(action_freq[AGENT_IDS.index(aid)].argmax())
        top_act     = ACTION_TYPES.get(top_act_idx, "?")
        print(f"  {STATE_AGENTS[aid]:<22} {survival_rate[aid]:>9.1%} "
              f"{defection_rate[aid]:>11.2f} {top_act:<20}")
    print(f"{'─'*60}\n")

    return {
        "action_freq":    action_freq,
        "alliance_freq":  alliance_freq,
        "survival_rate":  survival_rate,
        "defection_rate": defection_rate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    CSV_PATH = "worldsim_merged.csv"   # update for your Kaggle path

    # ── Quick smoke test (5 episodes, short cycles) ───────────────────────────
    print("\n[0] Smoke test (5 episodes) ...")
    cfg_test = MAPPOConfig(
        csv_path    = CSV_PATH,
        n_episodes  = 5,
        max_cycles  = 30,
        n_epochs    = 2,
        eval_every  = 5,
        save_every  = 99,
    )
    trainer = MAPPOTrainer(CSV_PATH, config=cfg_test)

    # Validate networks
    dummy_obs    = torch.zeros(1, cfg_test.obs_dim,             device=DEVICE)
    dummy_glob   = torch.zeros(1, cfg_test.obs_dim * cfg_test.n_agents, device=DEVICE)
    dummy_aid    = torch.zeros(1, dtype=torch.long,             device=DEVICE)
    act_l, tgt_l = trainer.actor(dummy_obs, dummy_aid)
    val          = trainer.critic(dummy_glob)
    assert act_l.shape == (1, N_ACTIONS),  f"Actor shape: {act_l.shape}"
    assert tgt_l.shape == (1, 10),          f"Target shape: {tgt_l.shape}"
    assert val.shape   == (1, 1),           f"Critic shape: {val.shape}"
    print("  ✓ Network forward pass shapes correct")

    log_df = trainer.train(n_episodes=5)
    assert len(log_df) == 5, "Expected 5 log rows"
    assert "mean_return" in log_df.columns
    assert "policy_loss" in log_df.columns
    print("  ✓ Training loop OK")

    # ── Full training ─────────────────────────────────────────────────────────
    print("\n[1] Full training run ...")
    cfg_full = MAPPOConfig(
        csv_path        = CSV_PATH,
        n_episodes      = 6000,
        max_cycles      = 150,
        n_epochs        = 4,  # Reduced
        minibatch_size  = 128,  # Increased
        eval_every      = 50,
        save_every      = 100,
        actor_hidden    = [256, 256],
        critic_hidden   = [512, 512],
        lr_actor        = 5e-5,  # Reduced further
        lr_critic       = 1e-4,  # Reduced further
        gamma           = 0.995,  # Increased
        gae_lambda      = 0.95,
        clip_eps        = 0.15,  # Reduced
        entropy_coeff   = 0.02,  # Adjusted
        value_coeff     = 1.0,  # Increased
        max_grad_norm   = 0.5,  # Tightened
        lr_decay        = 0.999,  # Slower
        entropy_start   = 0.1,  # Increased
        entropy_end     = 0.01,  # Increased
        curriculum_start= 0.70,
        curriculum_end  = 0.20,
    )
    trainer = MAPPOTrainer(CSV_PATH, config=cfg_full)
    log_df = trainer.train()

    # ── Training curves ───────────────────────────────────────────────────────
    print("\n[2] Generating training diagnostic plots ...")
    plot_training_curves(log_df, save_path="worldsim_training_curves.png")

    # ── Strategy analysis ─────────────────────────────────────────────────────
    print("\n[3] Emergent strategy analysis ...")
    # Load best checkpoint for analysis
    best_ckpt = os.path.join(cfg_full.checkpoint_dir,
                             "worldsim_india_ep0_best.pt")  # placeholder
    ckpts = [f for f in os.listdir(cfg_full.checkpoint_dir) if "best" in f]
    if ckpts:
        trainer.load_checkpoint(
            os.path.join(cfg_full.checkpoint_dir, sorted(ckpts)[-1]))
    analysis = analyse_emergent_strategies(
        trainer, n_eval_episodes=20,
        save_path="worldsim_strategy_analysis.png")

    # ── Save training log ─────────────────────────────────────────────────────
    log_df.to_csv("worldsim_training_log.csv", index=False)
    print("\n[4] Training log saved → worldsim_training_log.csv")

    print("\n✅ WorldSim India MAPPO complete.")
    print("   Outputs: worldsim_training_curves.png  |  "
          "worldsim_strategy_analysis.png  |  worldsim_training_log.csv")
