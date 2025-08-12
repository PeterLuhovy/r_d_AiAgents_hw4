

import os
import numpy as np
import matplotlib.pyplot as plt


def rollout_trajectory(env, agent=None, steps=200, greedy=False):
    """
    Vráti numpy polia (xs, ys, thetas) z jedného behu env.
    - agent=None => náhodné akcie
    - greedy=True => dočasne epsilon=0
    """
    obs, _ = env.reset()
    xs, ys, thetas = [], [], []

    old_eps = None
    if agent is not None and greedy and hasattr(agent, "curiosity"):
        old_eps = agent.curiosity
        agent.curiosity = 0.0

    done = False
    t = 0
    while not done and t < steps:
        x, y, vx, vy, theta, ang_v = obs
        xs.append(float(x)); ys.append(float(y)); thetas.append(float(theta))

        if agent is None:
            action = env.action_space.sample()
        else:
            s_idx = agent.discretize_state(obs)
            action = agent.choose_action(s_idx)

        obs, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        t += 1

    if old_eps is not None:
        agent.curiosity = old_eps

    return np.array(xs), np.array(ys), np.array(thetas)

def plot_trajectory(xs, ys, thetas=None, title="Trajektória hexapoda", out_path=None):
    """
    Nakreslí 2D dráhu + (voliteľne) šípky orientácie a uloží do out_path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys)
    ax.plot([xs[-1]], [ys[-1]], marker='o', markersize=5)  # aktuálna pozícia

    if thetas is not None and len(thetas) == len(xs):
        k = max(1, len(xs)//20)  # ~20 šípok
        u = np.cos(thetas[::k]) * 0.8
        v = np.sin(thetas[::k]) * 0.8
        ax.quiver(xs[::k], ys[::k], u, v, angles='xy', scale_units='xy', scale=1)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(title)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=160, bbox_inches='tight')
        print(f"✓ Trajektória uložená: {out_path}")
    else:
        plt.show()
    plt.close(fig)

def draw_and_save_trajectory(env, agent, log_dir, steps=200, greedy=True):
    """
    Urobí rollout a uloží PNG do logs/<run>/trajectory.png.
    """
    xs, ys, thetas = rollout_trajectory(env, agent=agent, steps=steps, greedy=greedy)
    out_path = os.path.join(log_dir, "trajectory.png")
    plot_trajectory(xs, ys, thetas, title="Hexapod – trajektória", out_path=out_path)
    # voliteľne ulož aj surové dáta
    np.savez_compressed(os.path.join(log_dir, "trajectory.npz"),
                        xs=xs.astype(np.float32), ys=ys.astype(np.float32), thetas=thetas.astype(np.float32))