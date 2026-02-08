def test_policy(env, policy, episodes=100):
    wins = 0
    print("\n--- Testing Value Iteration Policy ---")
    
    for i in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done:
            #  Force state to tuple of ints
            s = tuple(int(round(x)) for x in state)
            
            # Debug: Check if state is in policy
            action = policy.get(s)
            if action is None:
                print(f" Policy not defined for state {s}")
                action = np.random.randint(0, 4)  # Fallback
            else:
                if i < 3 and steps == 0:
                    act_names = ["Up", "Down", "Left", "Right"]
                    print(f"Episode {i+1}: Start {s} → Action {action} ({act_names[action]})")

            #  Take action
            state, reward, done = env.step(action)
            steps += 1

            if env.won:
                wins += 1
                if i < 3:
                    print(f"🎉 Won in {steps} steps!")
            elif env.game_over:
                if i < 3:
                    print(f"💥 Lost after {steps} steps.")

    win_rate = wins / episodes
    print(f"\n Value Iteration Win Rate: {win_rate:.2f} ({wins}/{episodes})")
    return win_rate