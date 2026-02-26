"""
WorldSim — MAPPO + Opponent Learning Training (FIXED v2)
=========================================================

ROOT CAUSE OF ZERO REWARDS:
  The env's step() method calls _agent_selector.next() TWICE:
      self._agent_selector.next()
      self.agent_selection = self._agent_selector.next()  # double advance!

  This means the env only processes every other agent per .step() call,
  so _world_step() (which assigns rewards) fires every 5 trainer steps,
  not 10. The old trainer looped over all 10 agents then read rewards —
  but rewards were already set (and reset) multiple times mid-loop.

THE FIX:
  Don't loop over trainer's own agent list. Instead:
    1. Read env.agent_selection to know whose turn it is
    2. Act for that agent
    3. Call env.step()
    4. Track env._cycle to detect when _world_step fired (cycle incremented)
    5. ONLY THEN read env.rewards for all agents

  This is the correct PettingZoo AEC pattern regardless of internal
  selector bugs — always let the env drive turn order.
"""

import subprocess, sys, os

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for _pkg in ["torch"]:
    try: __import__(_pkg)
    except ImportError: _install(_pkg)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[WorldSim MAPPO] Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

CSV_PATH = "/kaggle/input/worldsim/worldsim_final.csv"

try:
    _ = WorldSimEnv; _ = N_ACTIONS; _ = ACTION_TYPES
    print(f"[WorldSim MAPPO] Env symbols found — N_ACTIONS={N_ACTIONS}")
except NameError as _e:
    raise NameError(f"Run the env cell first: {_e}") from _e


# =====================================================================
# SECTION 1 — HYPERPARAMETERS
# =====================================================================

class MAPPOConfig:
    csv_path       = CSV_PATH
    n_agents       = 10
    obs_dim        = 78
    act_dim        = N_ACTIONS
    n_targets      = 10
    max_cycles     = 200
    noise_level    = 0.3

    n_episodes     = 3000
    rollout_length = 400      # total env.step() calls per rollout
    n_epochs       = 10
    batch_size     = 256
    gamma          = 0.995
    gae_lambda     = 0.95
    clip_eps       = 0.2
    entropy_coef   = 0.02
    value_coef     = 0.5
    max_grad_norm  = 0.5
    lr_actor       = 3e-4
    lr_critic      = 1e-3
    lr_opp_model   = 1e-3
    lr_min         = 1e-5
    reward_norm    = True

    hidden_dim     = 256
    gnn_hidden     = 128
    lstm_hidden    = 128
    n_top_rivals   = 3

    opp_bc_weight  = 0.3
    save_dir       = "/kaggle/working/worldsim_ckpt"
    save_freq      = 100

CFG = MAPPOConfig()
os.makedirs(CFG.save_dir, exist_ok=True)


# =====================================================================
# SECTION 2 — NEURAL NETWORKS
# =====================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class GNNEncoder(nn.Module):
    def __init__(self, node_feat=4, hidden=128):
        super().__init__()
        self.agg  = nn.Sequential(nn.Linear(node_feat, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.comb = nn.Sequential(nn.Linear(hidden*2, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.self_proj = nn.Linear(node_feat, hidden)
    def forward(self, rival_feats, self_feat):
        return self.comb(torch.cat([self.agg(rival_feats).mean(1),
                                    self.self_proj(self_feat)], -1))


class OppLSTM(nn.Module):
    def __init__(self, obs_dim, act_dim, n_targets, lstm_h=128):
        super().__init__()
        self.lstm_h   = lstm_h
        self.encoder  = nn.Linear(obs_dim*2, lstm_h)
        self.lstm     = nn.LSTM(lstm_h, lstm_h, batch_first=True)
        self.head_act = nn.Linear(lstm_h, act_dim)
        self.head_tgt = nn.Linear(lstm_h, n_targets)
    def forward(self, own_seq, riv_seq, hidden=None):
        x   = torch.cat([own_seq, riv_seq], -1)
        enc = F.relu(self.encoder(x))
        out, hidden = self.lstm(enc, hidden)
        return self.head_act(out), self.head_tgt(out), hidden
    def init_hidden(self, B):
        return (torch.zeros(1,B,self.lstm_h,device=DEVICE),
                torch.zeros(1,B,self.lstm_h,device=DEVICE))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, n_targets, hidden, gnn_h, lstm_h, n_rivals):
        super().__init__()
        self.gnn      = GNNEncoder(4, gnn_h)
        self.trunk    = MLP(obs_dim - 40, hidden, hidden)
        self.opp_proj = nn.Linear(n_rivals * lstm_h, hidden // 2)
        self.fusion   = MLP(hidden + gnn_h + hidden//2, hidden, hidden)
        self.act_head = nn.Linear(hidden, act_dim)
        self.tgt_head = nn.Linear(hidden, n_targets)

    def forward(self, obs, opp_beliefs=None):
        B = obs.shape[0]
        own  = torch.cat([obs[:,:31], obs[:,71:]], -1)
        rivs = obs[:,31:71].view(B,10,4)
        sf   = obs[:,:4]
        gnn  = self.gnn(rivs, sf)
        trk  = self.trunk(own)
        opp  = F.relu(self.opp_proj(opp_beliefs)) if opp_beliefs is not None \
               else torch.zeros(B, CFG.hidden_dim//2, device=obs.device)
        f    = self.fusion(torch.cat([trk, gnn, opp], -1))
        return self.act_head(f), self.tgt_head(f)


class Critic(nn.Module):
    def __init__(self, n_agents, obs_dim, hidden):
        super().__init__()
        self.net = MLP(n_agents * obs_dim, hidden*2, 1, n_layers=3)
    def forward(self, g): return self.net(g)


# =====================================================================
# SECTION 3 — AGENT
# =====================================================================

class MAPPOAgent:
    def __init__(self, agent_id, all_agents, cfg):
        self.agent_id = agent_id
        self.cfg      = cfg
        self.actor    = Actor(cfg.obs_dim, cfg.act_dim, cfg.n_targets,
                              cfg.hidden_dim, cfg.gnn_hidden, cfg.lstm_hidden,
                              cfg.n_top_rivals).to(DEVICE)
        self.rivals   = [a for a in all_agents if a != agent_id][:cfg.n_top_rivals]
        self.opps     = {r: OppLSTM(cfg.obs_dim, cfg.act_dim, cfg.n_targets,
                                     cfg.lstm_hidden).to(DEVICE)
                         for r in self.rivals}
        self.opp_h    = {}
        self.reset_hidden()

        self.opt_actor = Adam(self.actor.parameters(), lr=cfg.lr_actor)
        opp_params = sum([list(m.parameters()) for m in self.opps.values()], [])
        self.opt_opp   = Adam(opp_params, lr=cfg.lr_opp_model) if opp_params else None
        total_iters    = cfg.n_episodes * (cfg.rollout_length // cfg.n_agents)
        self.sched     = LinearLR(self.opt_actor, 1.0,
                                  cfg.lr_min/cfg.lr_actor, max(1, total_iters))

    def reset_hidden(self):
        for r in self.rivals:
            self.opp_h[r] = (torch.zeros(1,1,self.cfg.lstm_hidden,device=DEVICE),
                             torch.zeros(1,1,self.cfg.lstm_hidden,device=DEVICE))

    def get_beliefs(self):
        beliefs = [self.opp_h[r][0].squeeze() for r in self.rivals]
        while len(beliefs) < self.cfg.n_top_rivals:
            beliefs.append(torch.zeros(self.cfg.lstm_hidden, device=DEVICE))
        return torch.stack(beliefs).unsqueeze(0).view(1, -1)

    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        al, tl = self.actor(obs_t, self.get_beliefs())
        ad, td = Categorical(logits=al), Categorical(logits=tl)
        a, t   = ad.sample(), td.sample()
        return (np.array([a.item(), t.item()]),
                ad.log_prob(a).cpu().numpy(),
                td.log_prob(t).cpu().numpy())

    def update_opp(self, rival_id, own_t, riv_t):
        if rival_id not in self.opps: return
        _, _, h = self.opps[rival_id](own_t.unsqueeze(1), riv_t.unsqueeze(1),
                                       self.opp_h[rival_id])
        self.opp_h[rival_id] = (h[0].detach(), h[1].detach())


# =====================================================================
# SECTION 4 — BUFFER
# =====================================================================

class Buffer:
    def __init__(self): self.clear()

    def clear(self):
        self.obs = defaultdict(list); self.gobs  = defaultdict(list)
        self.act = defaultdict(list); self.alp   = defaultdict(list)
        self.tlp = defaultdict(list); self.rew   = defaultdict(list)
        self.don = defaultdict(list); self.val   = defaultdict(list)
        self.adv = defaultdict(list); self.ret   = defaultdict(list)

    def add(self, aid, obs, gobs, act, alp, tlp, rew, don, val):
        self.obs[aid].append(obs); self.gobs[aid].append(gobs)
        self.act[aid].append(act); self.alp[aid].append(alp)
        self.tlp[aid].append(tlp); self.rew[aid].append(rew)
        self.don[aid].append(don); self.val[aid].append(val)

    def finish(self, gamma, lam, last_vals):
        for aid in self.rew:
            R, V, D = self.rew[aid], self.val[aid], self.don[aid]
            T, adv, advs = len(R), 0.0, []
            lv = last_vals.get(aid, 0.0)
            for t in reversed(range(T)):
                nv   = lv if t == T-1 else V[t+1]
                mask = 1 - float(D[t])
                adv  = (R[t] + gamma*nv*mask - V[t]) + gamma*lam*mask*adv
                advs.insert(0, adv)
            self.adv[aid] = advs
            self.ret[aid] = [a+v for a,v in zip(advs, V)]

    def tensors(self, aid, dev):
        o  = torch.FloatTensor(np.array(self.obs[aid])).to(dev)
        go = torch.FloatTensor(np.array(self.gobs[aid])).to(dev)
        a  = torch.LongTensor(np.array(self.act[aid])[:,0]).to(dev)
        t  = torch.LongTensor(np.array(self.act[aid])[:,1]).to(dev)
        al = torch.FloatTensor(np.array(self.alp[aid])).to(dev)
        tl = torch.FloatTensor(np.array(self.tlp[aid])).to(dev)
        r  = torch.FloatTensor(np.array(self.ret[aid])).to(dev)
        adv= torch.FloatTensor(np.array(self.adv[aid])).to(dev)
        adv= (adv - adv.mean()) / (adv.std() + 1e-8)
        return o, go, a, t, al, tl, r, adv


# =====================================================================
# SECTION 5 — REWARD NORMALISER
# =====================================================================

class RNorm:
    def __init__(self):
        self.mean = self.var = np.float64(0); self.n = np.float64(1e-4)
    def update(self, x):
        b = float(x.size); tot = self.n + b; d = float(np.mean(x)) - self.mean
        self.mean += d*b/tot
        self.var   = (self.var*self.n + np.var(x)*b + d**2*self.n*b/tot) / tot
        self.n     = tot
    def norm(self, x):
        return float((x - self.mean) / (np.sqrt(self.var) + 1e-8))


# =====================================================================
# SECTION 6 — TRAINER  *** THE REAL FIX IS IN collect_rollout ***
# =====================================================================

class MAPPOTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = WorldSimEnv(csv_path=cfg.csv_path,
                               max_cycles=cfg.max_cycles,
                               noise_level=cfg.noise_level)
        self.agent_ids = self.env.possible_agents
        self.n         = len(self.agent_ids)
        self.agents    = {a: MAPPOAgent(a, self.agent_ids, cfg) for a in self.agent_ids}

        self.critic    = Critic(self.n, cfg.obs_dim, cfg.hidden_dim*2).to(DEVICE)
        self.opt_c     = Adam(self.critic.parameters(), lr=cfg.lr_critic)
        total_iters    = cfg.n_episodes * (cfg.rollout_length // self.n)
        self.sched_c   = LinearLR(self.opt_c, 1.0, cfg.lr_min/cfg.lr_critic,
                                  max(1, total_iters))
        self.rnorm     = {a: RNorm() for a in self.agent_ids}

        self.ep_rews   = defaultdict(list)
        self.a_losses  = []; self.c_losses = []; self.o_losses = []; self.entropies = []

        total_p = sum(p.numel() for ag in self.agents.values()
                      for p in list(ag.actor.parameters()) +
                      sum([list(m.parameters()) for m in ag.opps.values()],[]))
        print(f"[Trainer] {self.n} agents | Actor+Opp: {total_p:,} | "
              f"Critic: {sum(p.numel() for p in self.critic.parameters()):,}")

    # -----------------------------------------------------------------
    def _gobs(self, obs_d):
        return np.concatenate([obs_d.get(a, np.zeros(self.cfg.obs_dim, np.float32))
                                for a in self.agent_ids])

    @torch.no_grad()
    def _val(self, gobs):
        return self.critic(torch.FloatTensor(gobs).unsqueeze(0).to(DEVICE)).item()

    # -----------------------------------------------------------------
    def collect_rollout(self):
        """
        THE CORRECT AEC PATTERN:

        The env drives agent order via env.agent_selection.
        We must NOT loop over our own agent list — we follow the env.

        After env.step(), if env._cycle has incremented, _world_step fired,
        and env.rewards now contains valid rewards for all agents.

        We collect (obs, action, value) BEFORE env.step(),
        then collect rewards AFTER detecting a cycle completion.
        """
        buf = Buffer()
        for ag in self.agents.values(): ag.reset_hidden()

        obs_d, _ = self.env.reset(seed=None)
        ep_rews  = defaultdict(float)
        stored   = 0
        prev_obs = {}
        prev_cyc = self.env._cycle

        # Per-cycle staging: store transitions before reward is known
        staged: Dict[str, dict] = {}   # aid -> {obs, gobs, act, alp, tlp, val}

        while stored < self.cfg.rollout_length:
            # ── Follow the env's agent_selection ──────────────────────────
            aid = self.env.agent_selection

            # If env is done or agent is dead, call _was_dead_step and move on
            if (self.env.terminations.get(aid, False) or
                    self.env.truncations.get(aid, False)):
                try: self.env.step(None)
                except Exception: pass
                # Check if cycle completed
                if self.env._cycle != prev_cyc:
                    self._flush_staged(staged, buf, ep_rews, stored)
                    stored += len(staged)
                    staged.clear()
                    prev_cyc = self.env._cycle
                if not self.env.agents:
                    obs_d, _ = self.env.reset(seed=None)
                    for ag in self.agents.values(): ag.reset_hidden()
                    prev_obs.clear(); prev_cyc = self.env._cycle; staged.clear()
                continue

            # Update opponent models from previous cycle
            if prev_obs and aid in prev_obs:
                ag_obj = self.agents[aid]
                for riv in ag_obj.rivals:
                    if riv in prev_obs:
                        ot = torch.FloatTensor(prev_obs[aid]).unsqueeze(0).to(DEVICE)
                        rt = torch.FloatTensor(prev_obs[riv]).unsqueeze(0).to(DEVICE)
                        ag_obj.update_opp(riv, ot, rt)

            obs    = obs_d.get(aid, np.zeros(self.cfg.obs_dim, np.float32))
            gobs   = self._gobs(obs_d)
            val    = self._val(gobs)
            action, alp, tlp = self.agents[aid].act(obs)

            # Stage this transition — reward not known yet
            staged[aid] = dict(obs=obs.copy(), gobs=gobs, act=action,
                               alp=float(alp), tlp=float(tlp), val=val)

            # Step the env — this may trigger _world_step (reward assignment)
            try: self.env.step(action)
            except Exception: pass

            # Refresh obs
            if aid in self.env.agents:
                obs_d[aid] = self.env.observe(aid)
            else:
                obs_d[aid] = np.zeros(self.cfg.obs_dim, np.float32)

            # ── Detect cycle completion ─────────────────────────────────
            # _world_step fired when env._cycle incremented
            if self.env._cycle != prev_cyc:
                post_gobs = self._gobs(obs_d)
                n_flushed = self._flush_staged(staged, buf, ep_rews,
                                               stored, post_gobs)
                stored   += n_flushed
                prev_obs  = {k: v.copy() for k,v in obs_d.items()}
                staged.clear()
                prev_cyc  = self.env._cycle

                if not self.env.agents:
                    obs_d, _ = self.env.reset(seed=None)
                    for ag in self.agents.values(): ag.reset_hidden()
                    prev_obs.clear(); prev_cyc = self.env._cycle

        gobs_final = self._gobs(obs_d)
        lv = {a: self._val(gobs_final) if a in self.env.agents else 0.0
              for a in self.agent_ids}
        buf.finish(self.cfg.gamma, self.cfg.gae_lambda, lv)
        return buf, ep_rews

    def _flush_staged(self, staged, buf, ep_rews, already_stored, post_gobs=None):
        """Write staged transitions to buffer now that rewards are available."""
        count = 0
        for aid, s in staged.items():
            raw = self.env.rewards.get(aid, 0.0)
            if self.cfg.reward_norm:
                self.rnorm[aid].update(np.array([raw]))
                rew = self.rnorm[aid].norm(raw)
            else:
                rew = raw
            ep_rews[aid] += raw
            done = (self.env.terminations.get(aid, False) or
                    self.env.truncations.get(aid, False))
            gobs = post_gobs if post_gobs is not None else s['gobs']
            buf.add(aid, s['obs'], gobs, s['act'], s['alp'], s['tlp'],
                    rew, done, s['val'])
            count += 1
        return count

    # -----------------------------------------------------------------
    def ppo_update(self, buf):
        ta = tc = te = nc = 0
        for _ in range(self.cfg.n_epochs):
            for aid in self.agent_ids:
                if not buf.obs.get(aid): continue
                ag = self.agents[aid]
                o,go,a,t,oa,ot,r,adv = buf.tensors(aid, DEVICE)
                T = o.shape[0]
                if T == 0: continue
                idx = torch.randperm(T, device=DEVICE)
                for s in range(0, T, self.cfg.batch_size):
                    b   = idx[s:s+self.cfg.batch_size]
                    if not len(b): continue
                    al,tl = ag.actor(o[b], ag.get_beliefs().expand(len(b),-1))
                    ad,td = Categorical(logits=al), Categorical(logits=tl)
                    nal,ntl = ad.log_prob(a[b]), td.log_prob(t[b])
                    ent = (ad.entropy() + td.entropy()).mean()
                    rat = ((nal+ntl)-(oa[b]+ot[b])).exp()
                    s1,s2 = rat*adv[b], rat.clamp(1-self.cfg.clip_eps,
                                                    1+self.cfg.clip_eps)*adv[b]
                    al_ = -torch.min(s1,s2).mean() - self.cfg.entropy_coef*ent
                    cl_ = F.mse_loss(self.critic(go[b]).squeeze(-1), r[b]) * self.cfg.value_coef
                    loss = al_ + cl_
                    ag.opt_actor.zero_grad(); self.opt_c.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ag.actor.parameters(), self.cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                    ag.opt_actor.step(); self.opt_c.step()
                    ta += al_.item(); tc += cl_.item(); te += ent.item(); nc += 1
        d = max(1,nc)
        return ta/d, tc/d, te/d

    # -----------------------------------------------------------------
    def opp_update(self, buf):
        tl = n = 0
        for aid in self.agent_ids:
            ag = self.agents[aid]
            if not ag.opt_opp: continue
            for riv in ag.rivals:
                if not buf.obs.get(aid) or not buf.obs.get(riv): continue
                T = min(len(buf.obs[aid]), len(buf.obs[riv]))
                if T < 4: continue
                oo = torch.FloatTensor(np.array(buf.obs[aid][:T])).to(DEVICE)
                ro = torch.FloatTensor(np.array(buf.obs[riv][:T])).to(DEVICE)
                ra = torch.LongTensor(np.array(buf.act[riv][:T])[:,0]).to(DEVICE)
                rt = torch.LongTensor(np.array(buf.act[riv][:T])[:,1]).to(DEVICE)
                al,tl_,_ = ag.opps[riv](oo.unsqueeze(0), ro.unsqueeze(0),
                                         ag.opps[riv].init_hidden(1))
                loss = (F.cross_entropy(al.squeeze(0),ra) +
                        F.cross_entropy(tl_.squeeze(0),rt)) * self.cfg.opp_bc_weight
                ag.opt_opp.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(ag.opps[riv].parameters()),
                                         self.cfg.max_grad_norm)
                ag.opt_opp.step(); tl += loss.item(); n += 1
        return tl / max(1,n)

    # -----------------------------------------------------------------
    def save(self, ep):
        p = os.path.join(self.cfg.save_dir, f"ckpt_ep{ep:05d}.pt")
        ck = {"ep": ep, "critic": self.critic.state_dict(),
              "opt_c": self.opt_c.state_dict()}
        for aid,ag in self.agents.items():
            ck[f"{aid}_a"] = ag.actor.state_dict()
            ck[f"{aid}_oa"] = ag.opt_actor.state_dict()
            for r,m in ag.opps.items(): ck[f"{aid}_opp_{r}"] = m.state_dict()
        torch.save(ck, p); print(f"  [ckpt] -> {p}")

    def load(self, path):
        ck = torch.load(path, map_location=DEVICE)
        self.critic.load_state_dict(ck["critic"]); self.opt_c.load_state_dict(ck["opt_c"])
        for aid,ag in self.agents.items():
            if f"{aid}_a" in ck:
                ag.actor.load_state_dict(ck[f"{aid}_a"])
                ag.opt_actor.load_state_dict(ck[f"{aid}_oa"])
            for r in ag.rivals:
                if f"{aid}_opp_{r}" in ck: ag.opps[r].load_state_dict(ck[f"{aid}_opp_{r}"])
        print(f"[load] ep={ck['ep']}"); return ck["ep"]

    # -----------------------------------------------------------------
    def train(self, resume=None):
        start = self.load(resume) if resume else 0
        print(f"\n{'='*65}\n  WorldSim MAPPO — {self.cfg.n_episodes} episodes\n{'='*65}\n")
        t0 = time.time()

        for ep in range(start, self.cfg.n_episodes):
            buf, ep_rews = self.collect_rollout()

            if ep == 0:
                nz = sum(1 for v in ep_rews.values() if abs(v) > 1e-6)
                if nz == 0:
                    print("  *** STILL ZERO REWARDS — see debug below ***")
                    self._debug_rewards()
                else:
                    print(f"  Rewards flowing! nonzero agents: {nz}/10")

            al, cl, ent = self.ppo_update(buf)
            ol = self.opp_update(buf)

            for ag in self.agents.values(): ag.sched.step()
            self.sched_c.step()

            for aid in self.agent_ids:
                self.ep_rews[aid].append(ep_rews.get(aid, 0.0))
            self.a_losses.append(al); self.c_losses.append(cl)
            self.o_losses.append(ol); self.entropies.append(ent)

            if ep % 10 == 0:
                mr = np.mean([np.mean(v[-20:]) for v in self.ep_rews.values() if v])
                va = [a for a in self.ep_rews if self.ep_rews[a]]
                best  = max(va, key=lambda a: np.mean(self.ep_rews[a][-20:])) if va else "?"
                worst = min(va, key=lambda a: np.mean(self.ep_rews[a][-20:])) if va else "?"
                print(f"[EP {ep:>5d}/{self.cfg.n_episodes}]  "
                      f"MeanRew:{mr:>9.3f}  "
                      f"A:{al:>7.4f}  C:{cl:>7.4f}  "
                      f"O:{ol:>6.4f}  H:{ent:>5.3f}  "
                      f"t:{(time.time()-t0)/60:>5.1f}m  "
                      f"Best:{best}  Worst:{worst}")

            if ep % self.cfg.save_freq == 0 and ep > 0:
                self.save(ep)

        self.save(self.cfg.n_episodes)
        print(f"\nDone in {(time.time()-t0)/3600:.2f}h")
        return self._summary()

    def _debug_rewards(self):
        """Called when rewards are still zero — print diagnostic info."""
        print("\n  === REWARD DEBUG ===")
        print(f"  env._cycle after rollout: {self.env._cycle}")
        print(f"  env.rewards: {dict(list(self.env.rewards.items())[:3])}")
        # Test: manually force a world step
        print("  Testing manual _world_step()...")
        try:
            self.env._world_step()
            test_rew = {a: self.env._compute_reward(a) for a in self.env.agents[:3]}
            print(f"  Manual rewards: {test_rew}")
            if any(abs(v) > 1e-6 for v in test_rew.values()):
                print("  >> _world_step works but is_last() never fires in step()")
                print("  >> Check env.step() for double .next() bug")
        except Exception as e:
            print(f"  Manual _world_step failed: {e}")
        print("  ===================\n")

    def _summary(self):
        rows = [{"agent": a,
                 "mean_reward_final_100": np.mean(r[-100:]) if len(r)>=100 else np.mean(r),
                 "max": max(r), "min": min(r), "std": np.std(r)}
                for a,r in self.ep_rews.items() if r]
        df = pd.DataFrame(rows).sort_values("mean_reward_final_100", ascending=False)
        print("\nFinal performance:"); print(df.to_string(index=False))
        df.to_csv(f"{self.cfg.save_dir}/perf.csv", index=False)
        return df


# =====================================================================
# SECTION 7 — EVALUATION
# =====================================================================

def evaluate(trainer, n=20, render_every=5):
    print(f"\n[Eval] {n} episodes...")
    results = []
    for ep in range(n):
        obs_d, _ = trainer.env.reset(seed=ep)
        for ag in trainer.agents.values(): ag.reset_hidden()
        ep_rews = defaultdict(float); prev_cyc = trainer.env._cycle

        while trainer.env.agents:
            aid = trainer.env.agent_selection
            if (trainer.env.terminations.get(aid,False) or
                    trainer.env.truncations.get(aid,False)):
                try: trainer.env.step(None)
                except: pass
                continue
            obs = obs_d.get(aid, np.zeros(trainer.cfg.obs_dim, np.float32))
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                al,tl = trainer.agents[aid].actor(obs_t, trainer.agents[aid].get_beliefs())
            action = np.array([al.argmax().item(), tl.argmax().item()])
            try: trainer.env.step(action)
            except: pass
            if aid in trainer.env.agents:
                obs_d[aid] = trainer.env.observe(aid)
            if trainer.env._cycle != prev_cyc:
                for a in trainer.agent_ids:
                    ep_rews[a] += trainer.env.rewards.get(a, 0.0)
                prev_cyc = trainer.env._cycle

        if ep % render_every == 0: trainer.env.render()
        sdf = trainer.env.get_state_df()
        for _, row in sdf.iterrows():
            results.append({"ep": ep, "agent": row["iso3"],
                            "status": row.get("status","?"),
                            "reward": ep_rews.get(row["iso3"],0),
                            "water": row.get("water",0), "food": row.get("food",0)})

    df = pd.DataFrame(results)
    sur = df.groupby("agent")["status"].apply(lambda x:(x=="active").mean())\
            .reset_index().rename(columns={"status":"survival_rate"})
    mr  = df.groupby("agent")["reward"].mean().reset_index()
    s   = sur.merge(mr).sort_values("survival_rate", ascending=False)
    print(s.to_string(index=False))
    s.to_csv(f"{trainer.cfg.save_dir}/eval.csv", index=False)
    return df, s


def plot_curves(trainer):
    try:
        import matplotlib.pyplot as plt, matplotlib; matplotlib.use("Agg")
    except: return
    ISO_C = {"EGY":"#e74c3c","ETH":"#8e44ad","IND":"#f39c12","CHN":"#e91e63",
             "BRA":"#27ae60","DEU":"#2980b9","USA":"#2c3e50","SAU":"#f1c40f",
             "NGA":"#16a085","AUS":"#d35400"}
    fig, ax = plt.subplots(2,2,figsize=(14,8))
    fig.patch.set_facecolor("#0a0a0a")
    plt.suptitle("WorldSim MAPPO", color="white", fontsize=13, fontweight="bold")
    def sty(a,t,xl,yl):
        a.set_facecolor("#111111"); a.set_title(t,color="white",fontsize=9)
        a.set_xlabel(xl,color="white"); a.set_ylabel(yl,color="white")
        a.tick_params(colors="white"); a.spines[:].set_color("#333333")
        a.grid(True,alpha=0.12,color="white")
    w=20
    for aid,rews in trainer.ep_rews.items():
        if len(rews)>=w:
            ax[0,0].plot(pd.Series(rews).rolling(w).mean(),
                         color=ISO_C.get(aid,"#888"),lw=1.2,label=aid,alpha=0.9)
    sty(ax[0,0],f"Reward (roll {w})","Ep","Rew")
    ax[0,0].legend(fontsize=6,facecolor="#111",labelcolor="white",ncol=2)
    ax[0,1].plot(pd.Series(trainer.a_losses).rolling(w).mean(),color="#2ecc71",lw=1.2,label="Actor")
    ax[0,1].plot(pd.Series(trainer.c_losses).rolling(w).mean(),color="#e74c3c",lw=1.2,label="Critic")
    sty(ax[0,1],"Losses","Step","Loss")
    ax[0,1].legend(fontsize=7,facecolor="#111",labelcolor="white")
    ax[1,0].plot(pd.Series(trainer.entropies).rolling(w).mean(),color="#a78bfa",lw=1.2)
    sty(ax[1,0],"Entropy","Step","H")
    fm = {a:np.mean(r[-100:]) if len(r)>=100 else np.mean(r)
          for a,r in trainer.ep_rews.items() if r}
    sa = sorted(fm.items(),key=lambda x:x[1],reverse=True)
    ax[1,1].barh([x[0] for x in sa],[x[1] for x in sa],
                 color=[ISO_C.get(x[0],"#888") for x in sa])
    sty(ax[1,1],"Final Perf (100 ep)","Reward","Agent"); ax[1,1].invert_yaxis()
    plt.tight_layout(rect=[0,0,1,0.95])
    out = f"{CFG.save_dir}/curves.png"
    plt.savefig(out,dpi=120,bbox_inches="tight",facecolor="#0a0a0a"); plt.close()
    print(f"[Plot] -> {out}")


# =====================================================================
# SECTION 8 — EMERGENT BEHAVIOUR
# =====================================================================

def detect_patterns(env):
    sdf = env.get_state_df(); elog = env.get_event_log(); found = []
    checks = [
        ("upstream_cutoff",
         lambda: not elog.empty and
                 len(elog[elog["action"].isin(["reject_trade","defect_trade"])]) > 5),
        ("hegemon_isolation",
         lambda: not sdf.empty and "water" in sdf and sdf["water"].sum() > 0 and
                 sdf["water"].max()/sdf["water"].sum() > 0.4),
        ("commons_spiral",
         lambda: not sdf.empty and "water" in sdf and sdf["water"].mean() < 0.25 and
                 "n_allies" in sdf and (sdf["n_allies"]==0).all()),
        ("sustainable_coalition",
         lambda: not sdf.empty and "n_allies" in sdf and
                 len(sdf[sdf["n_allies"]>0])>=2 and len(sdf[sdf["n_allies"]==0])>=1 and
                 sdf[sdf["n_allies"]>0]["water"].mean()>0.35 and
                 sdf[sdf["n_allies"]==0]["water"].mean()<0.20),
    ]
    for name, fn in checks:
        try:
            if fn(): found.append(name); print(f"  PATTERN: {name}")
        except: pass
    return found


# =====================================================================
# SECTION 9 — MAIN
# =====================================================================

def main():
    print("\n" + "="*65)
    print("  WorldSim MAPPO v2 (fixed AEC rollout)")
    print("="*65)

    if torch.cuda.is_available():
        free = (torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_allocated()) / 1e9
        print(f"  Free VRAM: {free:.1f} GB")
        if free < 4.0:
            CFG.batch_size=128; CFG.hidden_dim=128; CFG.gnn_hidden=64; CFG.lstm_hidden=64

    trainer = MAPPOTrainer(CFG)

    # Reward sanity check
    print("\n[1/4] Sanity check...")
    buf, ep_rews = trainer.collect_rollout()
    total = sum(len(v) for v in buf.obs.values())
    print(f"  Transitions: {total}")
    nz_agents = 0
    for aid in trainer.agent_ids[:5]:
        rews = buf.rew.get(aid, [])
        if not rews: continue
        nz = sum(1 for r in rews if abs(r)>1e-6)
        print(f"  {aid}: min={min(rews):.4f} max={max(rews):.4f} "
              f"nonzero={nz}/{len(rews)} ep_total={ep_rews.get(aid,0):.3f}")
        if nz > 0: nz_agents += 1
    print("  >>> REWARDS OK <<<" if nz_agents > 0 else
          "  >>> REWARDS STILL ZERO — check env _world_step / is_last <<<")

    print(f"\n[2/4] Training {CFG.n_episodes} episodes...")
    summary = trainer.train()

    print("\n[3/4] Evaluating...")
    eval_df, eval_s = evaluate(trainer, n=20)

    print("\n[4/4] Plotting...")
    plot_curves(trainer)

    print("\n[+] Pattern detection...")
    trainer.env.reset(seed=99)
    for _ in range(50):
        for _ in range(trainer.n):
            if not trainer.env.agents: break
            try:
                trainer.env.step(np.array([np.random.randint(0,N_ACTIONS),
                                            np.random.randint(0,trainer.n)]))
            except: pass
    print(f"  Found: {detect_patterns(trainer.env) or ['none yet']}")

    print(f"\nDone. Checkpoints -> {CFG.save_dir}/")
    return trainer, summary, eval_df


if __name__ == "__main__":
    trainer, summary, eval_df = main()
