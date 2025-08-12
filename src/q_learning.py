import numpy as np

class QLearningAgent:
    def __init__(self, debug=False):
        """
        Vytvárame 'mozog' robota ktorý sa naučí chodiť
        """
        self.debug = debug
        
        # "OSOBNOSŤ" robota - ako sa správa:
        self.curiosity = 0.9        # epsilon *(zvedavosť)*
        self.learning_speed = 0.1   # learning rate *(rýchlosť učenia)*
        self.memory_decay = 0.99    # gamma *(ako dôležité sú budúce výsledky)*
        
        # DISKRETIZÁCIA PARAMETROV *(nastavenie pre rozdelenie situácií)*
        self.bins_per_dimension = 10  # Koľko "košíkov" pre každú hodnotu stavu
        
        # HRANICE STAVOVÉHO PRIESTORU *(min/max hodnoty z hexapod_env.py)*
        self.state_bounds = {
            'x_pos': (-20, 20),      # x pozícia v metroch
            'y_pos': (-20, 20),      # y pozícia v metroch  
            'x_vel': (-1, 1),        # x rýchlosť v m/s
            'y_vel': (-1, 1),        # y rýchlosť v m/s
            'orientation': (-3.14, 3.14),    # orientácia v radiánoch
            'angular_vel': (-2, 2)   # uhlová rýchlosť v rad/s
        }
        
        # VÝPOČET VEĽKOSTI Q-TABUĽKY
        total_states = self.bins_per_dimension ** 6  # 6 dimenzií stavu
        self.q_table = np.zeros((total_states, 3))   # 3 akcie: vpred, vľavo, vpravo

        if self.debug:
            print("🤖 === INICIALIZÁCIA Q-LEARNING AGENTA ===")
            print(f"✅ Zvedavosť (epsilon): {self.curiosity}")
            print(f"✅ Rýchlosť učenia: {self.learning_speed}")
            print(f"✅ Pamäť decay (gamma): {self.memory_decay}")
            print(f"✅ Košíky na dimenziu: {self.bins_per_dimension}")
            print(f"✅ Q-tabuľka: {self.q_table.shape}, bunky: {self.q_table.size}")
            print("=" * 50)



    def discretize_state(self, continuous_state):
        """
        STATE DISCRETIZATION *(prevod plynulých čísel na kategórie)*
        """
        discrete_state = []
        state_names = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'orientation', 'angular_vel']
        
        for i, (state_name, value) in enumerate(zip(state_names, continuous_state)):
            min_val, max_val = self.state_bounds[state_name]
            
            # NORMALIZÁCIA *(prevod na rozsah 0-1)*
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0, min(1, normalized))  # Orezanie na 0-1
            
            # DISKRETIZÁCIA *(zaradenie do kategórie 0-9)*
            discrete_value = int(normalized * (self.bins_per_dimension - 1))
            discrete_state.append(discrete_value)
        
        # KONVERZIA NA INDEX *(jediné číslo pre celý stav)*
        state_index = 0
        for i, val in enumerate(discrete_state):
            state_index += val * (self.bins_per_dimension ** i)
            
        return state_index
    
    def update(self, s_idx, a, r, s_next_idx, terminated: bool):
        """
        Q-learning update s debug výpisom.
        s_idx      = index aktuálneho stavu
        a          = vykonaná akcia
        r          = odmena
        s_next_idx = index nasledujúceho stavu
        terminated = True, ak epizóda skončila
        """
        current_q = self.q_table[s_idx, a]

        if terminated:
            target = r
        else:
            best_next_q = np.max(self.q_table[s_next_idx])
            target = r + self.memory_decay * best_next_q

        td_error = target - current_q
        new_q = current_q + self.learning_speed * td_error
        self.q_table[s_idx, a] = new_q

        if self.debug:
            print(f"[UPDATE] s={s_idx} a={a} r={r:.4f} term={terminated}")
            if not terminated:
                print(f"  best_next_q={best_next_q:.4f}")
            print(f"  current_q={current_q:.4f} target={target:.4f} td_error={td_error:.4f} new_q={new_q:.4f}")


    def choose_action(self, state_idx):
        """
        Epsilon-greedy výber akcie:
        - s pravdepodobnosťou epsilon (self.curiosity) vyber náhodnú akciu (prieskum)
        - inak vyber akciu s najvyššou Q-hodnotou (exploatácia)
        """
        if np.random.rand() < self.curiosity:
            # náhodná akcia (prieskum)
            return np.random.randint(self.q_table.shape[1])
        else:
            # najlepšia akcia podľa Q-tabuľky (exploatácia)
            return int(np.argmax(self.q_table[state_idx]))


# Test diskretizácie
if __name__ == "__main__":
    agent = QLearningAgent()
    
    # Test s príkladným stavom
    test_state = [5.2, -3.1, 0.5, -0.2, 1.57, 0.8]  # príklad z prostredia
    discrete_index = agent.discretize_state(test_state)
    print(f"📊 Stav {test_state} → diskrétny index {discrete_index}")