import gymnasium as gym
import numpy as np

class LunarLanderEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env) 

    def set_nonstoch_state(self, state, env_state=None):
        initial_state = [0, 0, 0, 0, 0, 0, 0, 0] 
        self.env.state = np.array(initial_state, dtype=np.float32)

    def get_state(self):
        return self.state
    
    def get_env_state(self):
        return self.state
    
    def get_actions(self, x=None):
        return range(self.env.action_space.n)
    
    def check_done(self): 
        if hasattr(self.env, 'done'):
            return self.env.done
        # Check if the lander has landed or crashed
        lander_y, lander_vy, _, _, leg1, leg2, _, _ = self.env.state
        landed = leg1 and leg2 and abs(lander_vy) < 0.5
        crashed = lander_y < 0 and abs(lander_vy) > 2
        return landed or crashed
    
    def check_done(self, state):  
        # Check if the lander has landed or crashed
        lander_y, lander_vy, _, _, leg1, leg2, _, _ = state
        landed = leg1 and leg2 and abs(lander_vy) < 0.5
        crashed = lander_y < 0 and abs(lander_vy) > 2
        return landed or crashed
    
    def equal_states(self, state1, state2, tol=1e-3):
        return np.allclose(state1, state2, atol=tol)
    
    def realistic(self, state):
        return True
