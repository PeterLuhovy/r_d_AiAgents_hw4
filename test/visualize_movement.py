import matplotlib.pyplot as plt
import numpy as np
from hexapod_env import SimpleHexapodEnv

def visualize_robot_path():
    print("=== VIZUALIZÁCIA POHYBU ROBOTA ===")
    
    env = SimpleHexapodEnv()
    env.reset()
    
    # Ukladaj trajektóriu
    positions = [env.position.copy()]
    orientations = [env.orientation]
    actions_taken = []
    rewards = []
    
    # Simuluj pohyb s rôznymi akciami
    action_sequence = [0, 0, 0, 0, 0,  # 5x vpred
                      1, 1, 1,         # 3x vľavo
                      0, 0, 0, 0,      # 4x vpred
                      2, 2, 2, 2,      # 4x vpravo  
                      0, 0, 0, 0, 0]   # 5x vpred
    
    for action in action_sequence:
        obs, reward, terminated, truncated, info = env.step(action)
        
        positions.append(env.position.copy())
        orientations.append(env.orientation)
        actions_taken.append(action)
        rewards.append(reward)
        
        if terminated:
            print("Simulácia skončila predčasne!")
            break
    
    # Vytvor graf
    positions = np.array(positions)
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Trajektória
    plt.subplot(2, 2, 1)
    plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, marker='o', markersize=3)
    plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Štart', zorder=5)
    plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='Koniec', zorder=5)
    
    # Ukáž orientáciu na každom 5. bode
    for i in range(0, len(positions), 5):
        x, y = positions[i]
        angle = orientations[i]
        dx = 0.5 * np.cos(angle)
        dy = 0.5 * np.sin(angle)
        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    plt.title('Trajektória Robota')
    plt.xlabel('X pozícia [m]')
    plt.ylabel('Y pozícia [m]')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    # Subplot 2: Pozícia v čase
    plt.subplot(2, 2, 2)
    steps = range(len(positions))
    plt.plot(steps, positions[:, 0], 'b-', label='X pozícia')
    plt.plot(steps, positions[:, 1], 'r-', label='Y pozícia')
    plt.title('Pozícia v čase')
    plt.xlabel('Krok')
    plt.ylabel('Pozícia [m]')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Orientácia v čase  
    plt.subplot(2, 2, 3)
    plt.plot(steps[1:], orientations[1:], 'g-', linewidth=2)
    plt.title('Orientácia v čase')
    plt.xlabel('Krok')
    plt.ylabel('Orientácia [rad]')
    plt.grid(True)
    
    # Subplot 4: Rewards v čase
    plt.subplot(2, 2, 4)
    plt.plot(range(len(rewards)), rewards, 'purple', linewidth=2)
    plt.title('Rewards v čase')
    plt.xlabel('Krok') 
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Štatistiky
    print(f"Celková vzdialenosť: {np.linalg.norm(positions[-1] - positions[0]):.3f}m")
    print(f"Priemerný reward: {np.mean(rewards):.3f}")
    print(f"Celkový reward: {np.sum(rewards):.3f}")

if __name__ == "__main__":
    visualize_robot_path()
# This code visualizes the movement of a hexapod robot in a custom environment.
# It plots the trajectory, position over time, orientation, and rewards received during the simulation.