import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from hexapod_env import SimpleHexapodEnv

def animate_robot():
    env = SimpleHexapodEnv()
    env.reset()
    
    # Priprav data
    #action_sequence = [0]*10 + [1]*50 + [0]*100 + [2]*50 + [0]*10  # mix akcií
    action_sequence = [0]*190  + [1]*10 + [0]*100 
    
    positions = [env.position.copy()]
    orientations = [env.orientation]
    
    for action in action_sequence:
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(env.position.copy())
        orientations.append(env.orientation)
        if terminated:
            break
    
    positions = np.array(positions)
    
    # Setup animácie
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 10)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Hexapod Robot Animation')
    
    # Elementy na kreslenie
    path_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
    robot_point, = ax.plot([], [], 'ro', markersize=10)
    orientation_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    def animate(frame):
        if frame >= len(positions):
            return path_line, robot_point, orientation_arrow
        
        # Aktualizuj trajektóriu
        path_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
        
        # Aktualizuj robot pozíciu
        robot_point.set_data([positions[frame, 0]], [positions[frame, 1]])
        
        # Aktualizuj orientáciu
        x, y = positions[frame]
        angle = orientations[frame]
        dx = np.cos(angle)
        dy = np.sin(angle)
        orientation_arrow.set_position((x, y))
        orientation_arrow.xy = (x + dx, y + dy)
        
        return path_line, robot_point, orientation_arrow
    
    anim = animation.FuncAnimation(fig, animate, frames=len(positions)+5,
                                 interval=200, blit=False, repeat=True)
    
    plt.show()

if __name__ == "__main__":
    animate_robot()