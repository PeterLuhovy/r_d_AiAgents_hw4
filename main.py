import numpy as np
import os, time, json
from src.hexapod_env import SimpleHexapodEnv
from src.q_learning import QLearningAgent


def train(env, agent, episodes=500, log_every=20):
    rewards_hist = []

    for ep in range(episodes):
        obs, _ = env.reset()
        s_idx = agent.discretize_state(obs)
        done = False
        ep_reward = 0.0

        while not done:
            # 1. Vyber akciu
            action = agent.choose_action(s_idx)

            # 2. Krok v prostredí
            obs_next, reward, terminated, truncated, _ = env.step(action)
            s_next_idx = agent.discretize_state(obs_next)

            # 3. Update Q-tabuľky
            agent.update(s_idx, action, reward, s_next_idx, terminated)

            # 4. Presun do ďalšieho stavu
            s_idx = s_next_idx
            ep_reward += reward
            done = terminated or truncated

        # 5. Decay epsilon (curiosity)
        agent.curiosity = max(0.05, agent.curiosity * 0.995)

        # 6. Logovanie odmien
        rewards_hist.append(ep_reward)
        if (ep + 1) % log_every == 0:
            avg = float(np.mean(rewards_hist[-log_every:]))
            print(f"Ep {ep+1}/{episodes} | avg_reward({log_every})={avg:.3f} | eps={agent.curiosity:.3f}")

    return rewards_hist

def save_q_table(agent, rewards_hist, out_dir="runs"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"qlearn_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # 1) Q-tab (komprimovane)
    np.savez_compressed(
        os.path.join(run_dir, "q_table.npz"),
        q_table=agent.q_table.astype(np.float32)
    )

    # 2) Meta info (hyperparametre, bins, rozmery)
    meta = {
      "bins_per_dimension": int(agent.bins_per_dimension),
      "n_actions": int(agent.q_table.shape[1]),
      "alpha": float(agent.learning_speed),
      "gamma": float(agent.memory_decay),
      "epsilon_final": float(agent.curiosity),
      "total_states": int(agent.q_table.shape[0]),
      "rewards_mean_last_20": float(np.mean(rewards_hist[-20:])) if len(rewards_hist)>=20 else float(np.mean(rewards_hist)),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 3) Rewardy (na rýchly graf neskôr)
    np.save(os.path.join(run_dir, "rewards.npy"), np.array(rewards_hist, dtype=np.float32))

    print(f"✓ Uložené do: {run_dir}")


if __name__ == "__main__":
    env = SimpleHexapodEnv(debug=False)
    agent = QLearningAgent(debug=False)

    rewards = train(env, agent, episodes=500, log_every=20)

    # === ULOŽENIE DO PRIEČINKA logs ===
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # uloženie Q-tabuľky
    np.savez_compressed(os.path.join(log_dir, "q_table.npz"), q_table=agent.q_table.astype(np.float32))

    # uloženie odmien
    np.save(os.path.join(log_dir, "rewards.npy"), np.array(rewards, dtype=np.float32))

    # uloženie meta údajov
    meta = {
        "episodes": 500,
        "bins_per_dimension": agent.bins_per_dimension,
        "n_actions": agent.q_table.shape[1],
        "alpha": agent.learning_speed,
        "gamma": agent.memory_decay,
        "epsilon_final": agent.curiosity,
        "total_states": agent.q_table.shape[0],
        "avg_reward_last_20": float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards))
    }
    with open(os.path.join(log_dir, "meta.txt"), "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    print(f"✓ Tréning hotový, všetko uložené do: {log_dir}")
