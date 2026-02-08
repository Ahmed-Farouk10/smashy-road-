# play_vi.py
import json
from env import SmashyRoadEnv

# Load policy
with open("results/models/value_iteration_policy.json", "r") as f:
    policy_str = json.load(f)
    policy = {eval(k): v for k, v in policy_str.items()}

env = SmashyRoadEnv(grid_size=10)

for episode in range(5):
    print(f"\n--- Playing Episode {episode+1} ---")
    state = env.reset()
    done = False
    steps = 0
    while not done:
        s = tuple(int(round(x)) for x in state)
        action = policy.get(s, 0)
        state, _, done = env.step(action)
        steps += 1
        env.render()
        if env.won:
            print(f"🎉 Won in {steps} steps!")
        elif env.game_over:
            print(f"💥 Lost at step {steps}")
    env.clock.tick(2)  # Slow down between episodes