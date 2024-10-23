def human_reward_function(state, ind, features, human_fun):
    agent_feats = features[ind, :]
    return human_fun(agent_feats, state)
