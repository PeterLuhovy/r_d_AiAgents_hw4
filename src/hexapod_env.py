

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
        lowParams[0] = -10      # x_pozícia minimum [m]
        lowParams[1] = -10      # y_pozícia minimum [m]  
        lowParams[2] = -1       # x_rýchlosť minimum [m/s]
        lowParams[3] = -1       # y_rýchlosť minimum [m/s]
        lowParams[4] = -np.pi   # orientácia minimum [rad]
        lowParams[5] = -2       # uhlová rýchlosť minimum [rad/s]

        highParams = [0] * 6
        highParams[0] = 10      # x_pozícia maximum  [m]
        highParams[1] = 10      # y_pozícia maximum   [m]
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
        self.max_steps = 20000  # Maximálny počet krokov v epizóde
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
    
    def _dist_to_north_edge(self, pos=None):
        # vzdialenosť k severnej hrane (y=+20), clamp na >=0
        y = (self.position[1] if pos is None else pos[1])
        return max(0.0, 20.0 - float(y))

    def _reached_north_edge(self, tol=0.05):
        return self.position[1] >= 20.0 - tol
    
    def _wrap_angle(self, angle):
        """Zabezpečí že uhol zostane medzi -π a +π"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def _calculate_reward(self, success=False):
        """
        Brutálne sever:
        + silný bonus za heading k +Y
        + odmena za Δy, ALE škrtaná bočným driftom
        - tvrdá penalizácia za |x| a |vx| (držať koridor)
        - časový trest
        + terminálny bonus pri dotyku severu
        """
        # progres iba na sever
        y_now = float(self.position[1])
        dy = y_now - self.prev_y                 # >0 ak ideš hore

        # heading k severu (target = +Y = pi/2)
        angle_err = self.orientation - (np.pi/2)
        heading_term = np.cos(angle_err)         # [-1,1]
        heading_bonus = 0.3 * heading_term       # silný tlak na natočenie

        # bočný drift a vybočenie
        vx = float(self.velocity[0])
        side_drift_pen = 0.15 * abs(vx)          # trest za bočnú rýchlosť
        corridor_pen   = 0.02 * abs(self.position[0])  # trest za |x| od stredovej osi

        # časový trest
        time_penalty = 0.02

        # odmena za Δy, ale „škrtáme“ ju pri bočnom drifte:
        dy_effective = max(0.0, dy) * max(0.0, 1.0 - min(1.0, 2.0*abs(vx)))  # keď bočíš, Δy sa zmenší
        dy_reward = 8.0 * dy_effective

        reward = dy_reward + heading_bonus - side_drift_pen - corridor_pen - time_penalty

        if success:
            reward += 12.0  # veľký koncový bonus

        # update pre ďalší krok
        self.prev_y = y_now
        return float(reward)

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        if self.debug:
            print("Resetujem hexapod environment!")
        self.position = np.array([
            np.random.uniform(-8.0, 8.0),   # x
            np.random.uniform(-8.0, 8.0)    # y
            ], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.orientation = float(np.random.uniform(-np.pi, np.pi))
        self.angular_velocity = 0.0  # uhlová rýchlosť v radiánoch za sekundu
        self.current_step = 0  # Počítadlo krokov
        self.prev_dist_to_north = self._dist_to_north_edge()
        self.prev_y = float(self.position[1])
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
        success = self._reached_north_edge()
        
        # 5. Create observation
        observation = self._get_observation()
        
        # 6. Calculate reward
        
        reward = self._calculate_reward(success=success)
        
        # 7. Check termination
        terminated = False
        terminated = success or (self.current_step >= self.max_steps) \
             or (abs(self.position[0]) > 20) or (abs(self.position[1]) > 20 + 1e-6)
        
        # 8. Truncated and info
        truncated = False
        info = {}



        # 9. Return observation, reward, terminated, truncated, info
        if self.debug:
            print(f"Position: {self.position}, Velocity: {self.velocity}, Orientation: {self.orientation}, Angular Velocity: {self.angular_velocity}")
            print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        return observation, reward, terminated, truncated, info

