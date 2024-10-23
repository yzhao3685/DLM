def base_reward_function(state, ind, features, base_fun):
    agent_feats = features[ind, :]
    return base_fun(agent_feats, state)
