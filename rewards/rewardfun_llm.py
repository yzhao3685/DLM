def llm_reward_function(state, ind, features):
    agent_feats = features[ind, :]
    return  state and agent_feats[9] * 3 + state and agent_feats[10] * 2 + (state and agent_feats[11]) * 5 
