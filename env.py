# env.py
import json
import numpy as np
import pygame

class SmashyRoadEnv:
    """
    2D Police Chase Game Environment
    - Agent must reach goal (9,9)
    - Avoid police (greedy follower)
    - Fuel gives +5 reward
    - -1 per step, -100 on capture, +100 on goal
    - Obstacles block movement
    """

    def __init__(self, grid_size=10, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.scale = 40
        self.width = grid_size * self.scale
        self.height = grid_size * self.scale
        self.action_space_size = 4  # Up, Down, Left, Right

        # Positions
        self.agent_pos = [2, 2]
        self.police_pos = [7, 7]
        self.goal_pos = [grid_size - 1, grid_size - 1]  # (9,9)
        self.fuel_pos = [5, 5]
        self.fuel_collected = False

        # Obstacles (buildings)
        self.buildings = [
            [3, 3], [3, 4], [4, 3],
            [1, 7], [2, 7], [1, 8],
            [6, 1], [7, 1], [6, 2]
        ]

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.screen = None
        self.clock = None
        self.font = None
        self.explosion = False
        self.explosion_time = 0

    def reset(self):
        self.agent_pos = [2, 2]
        self.police_pos = [7, 7]
        self.fuel_collected = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.explosion = False
        self.explosion_time = 0
        return self._get_state()

    def _get_state(self):
        return tuple(self.agent_pos + self.police_pos)

    def action_space_sample(self):
        return np.random.randint(0, 4)

    def step(self, action):
        ax, ay = self.agent_pos
        px, py = self.police_pos

        # Agent movement
        new_ax, new_ay = ax, ay
        if action == 0:   # Up
            new_ax = max(0, ax - 1)
        elif action == 1: # Down
            new_ax = min(self.grid_size - 1, ax + 1)
        elif action == 2: # Left
            new_ay = max(0, ay - 1)
        elif action == 3: # Right
            new_ay = min(self.grid_size - 1, ay + 1)

        # Only move if not in a building
        if [new_ax, new_ay] not in self.buildings:
            ax, ay = new_ax, new_ay
        self.agent_pos = [ax, ay]

        # ✅ CHECK WIN FIRST
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
            self.won = True
            return self._get_state(), reward, done

        # Police follows (with obstacle check)
        if px < ax and [px + 1, py] not in self.buildings:
            px += 1
        elif px > ax and [px - 1, py] not in self.buildings:
            px -= 1
        if py < ay and [px, py + 1] not in self.buildings:
            py += 1
        elif py > ay and [px, py - 1] not in self.buildings:
            py -= 1
        self.police_pos = [px, py]

        self.steps += 1
        reward = -1  # Step penalty
        done = False

        # ✅ THEN CHECK POLICE COLLISION
        if self.agent_pos == self.police_pos:
            reward = -100
            done = True
            self.game_over = True
            self.explosion = True
            self.explosion_time = 60
            return self._get_state(), reward, done

        # Fuel collected
        if not self.fuel_collected and self.agent_pos == self.fuel_pos:
            reward = 5
            self.score += 5
            self.fuel_collected = True

        # Max steps
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Police Catcher Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill((10, 10, 10))

        # Grid lines
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.scale, y * self.scale, self.scale, self.scale)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

        # Buildings
        for bx, by in self.buildings:
            rect = pygame.Rect(bx * self.scale, by * self.scale, self.scale, self.scale)
            pygame.draw.rect(self.screen, (128, 0, 128), rect)

        # Goal
        gx, gy = self.goal_pos
        goal_rect = pygame.Rect(gy * self.scale, gx * self.scale, self.scale, self.scale)
        pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)

        # Fuel
        if not self.fuel_collected:
            fx, fy = self.fuel_pos
            fuel_center = (fy * self.scale + 20, fx * self.scale + 20)
            pygame.draw.circle(self.screen, (255, 255, 0), fuel_center, 10)

        # Explosion
        if self.explosion and self.explosion_time > 0:
            self.explosion_time -= 1
            ax, ay = self.agent_pos
            center = (ay * self.scale + 20, ax * self.scale + 20)
            pygame.draw.circle(self.screen, (255, 0, 0), center, 30, 3)
            if self.explosion_time < 30:
                pygame.draw.circle(self.screen, (255, 100, 0), center, 15, 2)

        # Agent (red)
        ax, ay = self.agent_pos
        agent_center = (ay * self.scale + 20, ax * self.scale + 20)
        pygame.draw.circle(self.screen, (255, 0, 0), agent_center, 12)

        # Police (blue)
        px, py = self.police_pos
        police_center = (py * self.scale + 20, px * self.scale + 20)
        pygame.draw.circle(self.screen, (0, 0, 255), police_center, 12)

        # HUD
        steps_text = self.font.render(f"Steps: {self.steps}", True, (255, 255, 255))
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(steps_text, (5, 5))
        self.screen.blit(score_text, (5, 30))

        pygame.display.flip()
        self.clock.tick(10)
        return True

    def save_game_state(self, filename):
        state = {
            'agent_pos': [int(x) for x in self.agent_pos],
            'police_pos': [int(x) for x in self.police_pos],
            'steps': int(self.steps),
            'score': int(self.score),
            'game_over': bool(self.game_over),
            'won': bool(self.won),
            'explosion': bool(self.explosion),
            'explosion_time': int(self.explosion_time),
            'fuel_collected': bool(self.fuel_collected)
        }
        try:
            with open(filename, 'w') as f:
                json.dump(state, f)
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False

    def load_game_state(self, filename):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            self.agent_pos = [int(x) for x in state['agent_pos']]
            self.police_pos = [int(x) for x in state['police_pos']]
            self.steps = int(state['steps'])
            self.score = int(state['score'])
            self.game_over = bool(state['game_over'])
            self.won = bool(state['won'])
            self.explosion = bool(state['explosion'])
            self.explosion_time = int(state['explosion_time'])
            self.fuel_collected = bool(state.get('fuel_collected', False))
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False