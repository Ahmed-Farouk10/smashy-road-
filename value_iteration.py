# value_iteration.py
import numpy as np
from env import SmashyRoadEnv

def value_iteration(env, gamma=0.9, theta=1e-6):
    N = env.grid_size

    # ✅ Generate ALL possible states
    states = [(ax, ay, px, py) for ax in range(N) for ay in range(N)
              for px in range(N) for py in range(N)]
    
    # Initialize value function
    V = {s: 0.0 for s in states}

    def get_next_state(s, a):
        ax, ay, px, py = s
        new_ax, new_ay = ax, ay

        # Agent movement
        if a == 0:   # Up
            new_ax = max(0, ax - 1)
        elif a == 1: # Down
            new_ax = min(N - 1, ax + 1)
        elif a == 2: # Left
            new_ay = max(0, ay - 1)
        elif a == 3: # Right
            new_ay = min(N - 1, ay + 1)

        # ✅ Block movement into buildings
        if [new_ax, new_ay] in env.buildings:
            new_ax, new_ay = ax, ay

        # Police follows greedily (with obstacle check)
        new_px, new_py = px, py
        if new_px < new_ax and [new_px + 1, new_py] not in env.buildings:
            new_px += 1
        elif new_px > new_ax and [new_px - 1, new_py] not in env.buildings:
            new_px -= 1
        if new_py < new_ay and [new_px, new_py + 1] not in env.buildings:
            new_py += 1
        elif new_py > new_ay and [new_px, new_py - 1] not in env.buildings:
            new_py -= 1

        return (new_ax, new_ay, new_px, new_py)

    def get_reward(s):
        ax, ay, px, py = s
        # ✅ Use list comparison to match env.goal_pos = [9,9]
        if [ax, ay] == env.goal_pos:
            return 100
        elif [ax, ay] == [px, py]:
            return -100
        else:
            return -1  # Step penalty

    # ✅ Value Iteration Loop
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        for s in list(V.keys()):
            v = V[s]
            best_q = -np.inf
            for a in range(4):
                ns = get_next_state(s, a)
                r = get_reward(ns)
                q = r + gamma * V.get(ns, 0)
                best_q = max(best_q, q)
            V[s] = best_q
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    print(f"✅ Value Iteration converged in {iteration} iterations.")

    # ✅ Build policy for every state
    policy = {}
    for s in states:
        best_a = 0
        best_q = -np.inf
        for a in range(4):
            ns = get_next_state(s, a)
            r = get_reward(ns)
            q = r + gamma * V.get(ns, 0)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a

    # ✅ Debug: Test policy near goal
    test_state = (9, 8, 0, 0)
    if test_state in policy:
        action = policy[test_state]
        print(f"✅ Policy test: At {test_state} → Action {action} (should be 3=Right)")
    else:
        print(f"❌ Policy missing test state {test_state}")

    return policy