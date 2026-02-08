# test_vi_full.py
from env import SmashyRoadEnv
from value_iteration import value_iteration

# Create environment
env = SmashyRoadEnv(grid_size=10)
print("Computing Value Iteration policy...")
policy = value_iteration(env)

# Run one full episode with full logging
print("\n🚀 Running full episode with Value Iteration policy...\n")
state = env.reset()
done = False
steps = 0
act_names = ["Up", "Down", "Left", "Right"]

while not done:
    ax, ay, px, py = [int(round(x)) for x in state]
    ax = max(0, min(9, ax))
    ay = max(0, min(9, ay))
    px = max(0, min(9, px))
    py = max(0, min(9, py))
    s = (ax, ay, px, py)

    action = policy.get(s)
    if action is None:
        print(f"❌ No policy for state {s}! Using random action.")
        action = 0

    print(f"Step {steps:2d}: Agent({ax},{ay}) | Police({px},{py}) | Action: {act_names[action]} ({action})")

    state, reward, done = env.step(action)
    steps += 1

    if env.won:
        print(f"🎉 WIN! Reached goal in {steps} steps.")
    elif env.game_over:
        print(f"💥 CAUGHT by police at step {steps}.")
    elif steps >= 100:
        print(f"⏰ Max steps reached at step {steps}.")
        break

if not env.won and not env.game_over:
    print("⏸️  Episode ended without win or loss.")