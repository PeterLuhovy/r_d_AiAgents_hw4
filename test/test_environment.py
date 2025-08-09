# test_environment.py
from hexapod_env import SimpleHexapodEnv

env = SimpleHexapodEnv()

# Test reset
obs, info = env.reset()
print(f"Počiatočná observácia: {obs}")

# Test few steps
for i in range(100):
    action = env.action_space.sample()  # náhodná akcia
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Krok {i+1}: reward={reward:.3f}")
    
    if terminated:
        print("Epizóda skončila!")
        break