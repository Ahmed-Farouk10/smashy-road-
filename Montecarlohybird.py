def mcts_action(env, num_simulations=20, value_policy=None):
    root = MCTSNode(state=env._get_state())

    for _ in range(num_simulations):
        node = root
        temp_env = LightweightSmashyRoadEnv(grid_size=10)
        temp_env.agent_pos = env.agent_pos[:]
        temp_env.police_pos = env.police_pos[:]
        temp_env.fuel_collected = env.fuel_collected
        temp_env.steps = env.steps
        temp_env.game_over = env.game_over
        temp_env.won = env.won

        # Selection
        while node.untried_actions == [] and node.children:
            node = max(node.children, key=lambda n: n.ucb1())
            temp_env.step(node.action)

        # Expansion
        if node.untried_actions:
            # Use Value Iteration policy if available
            s = temp_env._get_state()
            ax, ay, px, py = [int(round(x)) for x in s]
            ax = np.clip(ax, 0, 9)
            ay = np.clip(ay, 0, 9)
            px = np.clip(px, 0, 9)
            py = np.clip(py, 0, 9)
            s_key = (ax, ay, px, py)

            action = value_policy.get(s_key, random.choice(node.untried_actions))
            if action not in node.untried_actions:
                action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)

            _, _, done = temp_env.step(action)
            child_state = temp_env._get_state()
            child = MCTSNode(state=child_state, parent=node, action=action)
            node.children.append(child)
            node = child

        # Simulation
        reward = simulate_random_playout(temp_env, max_steps=50)

        # Backpropagation
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent

    # Return best action by visit count
    if root.children:
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    else:
        return random.randint(0, 3)