

import gymnasium as gym
import numpy as np



class SimpleHexapodEnv(gym.Env):
    """
    Hexapod environment for reinforcement learning tasks.
    This environment simulates a hexapod robot with six legs.
    """

    def __init__(self, debug=False):
        self.debug = debug
        if self.debug:
            print("Vytváram nový hexapod environment!")

        super().__init__()
        
        lowParams = [0] * 6
        lowParams[0] = -20      # x_pozícia minimum [m]
        lowParams[1] = -20      # y_pozícia minimum [m]  
        lowParams[2] = -1       # x_rýchlosť minimum [m/s]
        lowParams[3] = -1       # y_rýchlosť minimum [m/s]
        lowParams[4] = -np.pi   # orientácia minimum [rad]
        lowParams[5] = -2       # uhlová rýchlosť minimum [rad/s]

        highParams = [0] * 6
        highParams[0] = 20      # x_pozícia maximum  [m]
        highParams[1] = 20      # y_pozícia maximum   [m]
        highParams[2] = 1       # x_rýchlosť maximum [m/s]
        highParams[3] = 1       # y_rýchlosť maximum [m/s]  
        highParams[4] = np.pi   # orientácia maximum [rad]
        highParams[5] = 2       # uhlová rýchlosť maximum [rad/s]

        self.observation_space = gym.spaces.Box(low=np.array(lowParams), high=np.array(highParams), shape=(6,), dtype=np.float32)

        # Q-learning (diskrétne akcie):
        self.action_space = gym.spaces.Discrete(3)     # 0 = vpred, 1 = vľavo, 2 = vpravo

        # pre PPO (kontinuálne akcie):
        # self.action_space = gym.spaces.Box(
        #    low=np.array([-1, -1]), 
        #    high=np.array([1, 1]), 
        #    dtype=np.float32
        #)
        # [sila_vpred, sila_otočenia] každá od -1 do +1

        self.position = np.array([0.0, 0.0])  # [x, y]
        self.velocity = np.array([0.0, 0.0])  # [vx, vy]
        self.orientation = 0.0  # orientácia v radiánoch
        self.angular_velocity = 0.0  # uhlová rýchlosť v radiánoch za sekundu
        self.max_steps = 200  # Maximálny počet krokov v epizóde
        self.current_step = 0  # Počítadlo krokov
        self.dt = 0.1  # Časový krok simulácie (0.1 sekundy)

    def _get_observation(self):
        # Return the current observation
        return np.concatenate([
            self.position,
            self.velocity,
            [self.orientation],
            [self.angular_velocity]
        ]).astype(np.float32)
    
    def _wrap_angle(self, angle):
        """Zabezpečí že uhol zostane medzi -π a +π"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def _calculate_reward(self):
        """Vypočíta odmenu na základe aktuálneho stavu"""
        reward = 0

        # A) POZITÍVNE REWARDS - čo odmeňuješ
        positiveReward = 0

        # Odmena za to že je ďaleko od začiatku
        distance_from_origin = np.sqrt(self.position[0]**2 + self.position[1]**2)
        positiveReward += distance_from_origin * 0.1  # koeficient 0.1

        reward += positiveReward
        # B) NEGATÍVNE REWARDS - čo trestáš  
        negativeReward = 0
        # Trest za rýchlosť mimo rozsahu
        if abs(self.velocity[0]) > 1 or abs(self.velocity[1]) > 1:
            negativeReward = 0.5  # koeficient 0.5
        reward -= negativeReward  # trest za rýchlosť mimo rozsahu

        # C) SHAPE REWARDS - jemné vedenie
        shapeReward = 0
        # Malá odmena za každý krok (motivácia prežiť)
        shapeReward += 1.0
        
        reward += shapeReward

        return reward

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        if self.debug:
            print("Resetujem hexapod environment!")
        self.position = np.array([0.0, 0.0])  # [x, y]
        self.velocity = np.array([0.0, 0.0])  # [vx, vy]
        self.orientation = 0.0  # orientácia v radiánoch
        self.angular_velocity = 0.0  # uhlová rýchlosť v radiánoch za sekundu
        self.current_step = 0  # Počítadlo krokov
        return self._get_observation(), {}

    def step(self, action):
        # 1. Update step counter
        self.current_step += 1
        if self.debug:
            print(f"Step: {self.current_step} ---> Action: {action}")
        
        # 2. Apply actions
        if action == 0:  # vpred
            # Pridaj silu v smere kde robot pozerá
            force = 0.1
            self.velocity[0] += force * np.cos(self.orientation)  # x komponenta
            self.velocity[1] += force * np.sin(self.orientation)  # y komponenta
        elif action == 1:  # vľavo
            self.angular_velocity += 0.1
        elif action == 2:  # vpravo
            self.angular_velocity -= 0.1
        
        # 3. Apply dampening
        self.velocity *= 0.9
        self.angular_velocity *= 0.8
        
        # 4. Update position and orientation (TOTO TI CHÝBALO!)
        self.position += self.velocity * self.dt
        self.orientation += self.angular_velocity * self.dt
        self.orientation = self._wrap_angle(self.orientation)  # Udržuj uhol medzi -π a +π
        
        # 5. Create observation
        observation = self._get_observation()
        
        # 6. Calculate reward
        reward = self._calculate_reward()
        
        # 7. Check termination
        terminated = False
        if self.current_step >= self.max_steps:
            terminated = True
        if abs(self.position[0]) > 20 or abs(self.position[1]) > 20:
            terminated = True  # robot vyšiel zo sveta
        
        # 8. Truncated and info
        truncated = False
        info = {}

        # 9. Return observation, reward, terminated, truncated, info
        if self.debug:
            print(f"Position: {self.position}, Velocity: {self.velocity}, Orientation: {self.orientation}, Angular Velocity: {self.angular_velocity}")
            print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        return observation, reward, terminated, truncated, info

