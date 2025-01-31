import torch
import numpy as np
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")
device = 'cpu' 


def do_eval(model, batchs, batch_size=128):
    (state, next_state, action, next_action, reward, done, bloc_num, SOFAS) = batchs
    Q_value = model.Q(state)
    agent_actions = torch.argmax(Q_value, dim=1)
    phy_actions = action
    Q_value_pro1 = F.softmax(Q_value)
    Q_value_pro_ind = torch.argmax(Q_value_pro1, dim=1)
    Q_value_pro_ind1 = range(len(Q_value_pro_ind))
    Q_value_pro = Q_value_pro1[Q_value_pro_ind1, Q_value_pro_ind]
    return Q_value, agent_actions, phy_actions, Q_value_pro

def do_eval_ac(agent, batches, batch_size=128):
    # Unpack the batch data
    states, next_states, actions, next_actions, rewards, done_flags, bloc_num, SOFAS = batches
    
    # Move states to the device
    states = states.to(agent.device)
    next_states = next_states.to(agent.device)
    
    # Get the value from the critic network (Q-value approximation)
    state_values = agent.critic(states)
    
    # Get the action probabilities from the actor network (policy)
    action_probs = agent.actor(states)
    
    # Select agent actions (greedy approach by taking the argmax of the policy)
    agent_actions = torch.argmax(action_probs, dim=1)
    
    # The physical actions come directly from the batch (assuming they are the ground truth actions)
    phy_actions = actions
    
    # Calculate the Q-values from the critic network for the next states
    with torch.no_grad():
        next_state_values = agent.critic_target(next_states)
    
    # Calculate the Q-values using the reward, the next state value, and whether the episode is done
    Q_value = rewards + agent.gamma * next_state_values * (1 - done_flags)
    
    # Softmax the Q-values for probabilistic representation
    Q_value_softmax = F.softmax(Q_value, dim=1)
    
    # Get the indices of the max Q-values
    Q_value_indices = torch.argmax(Q_value_softmax, dim=1)
    
    # Get the corresponding Q-value probabilities
    Q_value_probs = Q_value_softmax[range(len(Q_value_indices)), Q_value_indices]
    
    return Q_value, agent_actions, phy_actions, Q_value_probs


def do_test(save_path, model, Xtest, actionbloctest, bloctest, Y90, SOFA, reward_value, beat):
    bloc_max = max(bloctest)  # 最大才20个阶段
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
    R3 = r2[:, 0]
    R4 = (R3 + reward_value) / (2 * reward_value)
    RNNstate = Xtest
    print('####  生成测试集轨迹  ####')
    statesize = int(RNNstate.shape[1])
    states = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), statesize))
    actions = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1), dtype=int)
    next_actions = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1), dtype=int)
    rewards = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    next_states = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), statesize))
    done_flags = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    bloc_num = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    blocnum1 = 1
    c = 0

    bloc_num_reward = 0
    for i in range(RNNstate.shape[0] - 1):  # 每一行循环
        states[c] = RNNstate[i, :]
        actions[c] = actionbloctest[i]
        bloc_num[c] = blocnum1
        if (bloctest[i + 1] == 1):  # end of trace for this patient
            next_states1 = np.zeros(statesize)
            next_actions1 = -1
            done_flags1 = 1
            blocnum1 = blocnum1 + 1
            bloc_num_reward += 1
            reward1 = -beat[0] * (SOFA[i]) + R3[i]
            bloc_num_reward = 0
        else:
            next_states1 = RNNstate[i + 1, :]
            next_actions1 = actionbloctest[i + 1]
            done_flags1 = 0
            blocnum1 = blocnum1
            reward1 = - beat[1] * (SOFA[i + 1] - SOFA[i])
            bloc_num_reward += 1
        next_states[c] = next_states1
        next_actions[c] = next_actions1
        rewards[c] = reward1
        done_flags[c] = done_flags1
        c = c + 1  # 从0开始
    states[c] = RNNstate[c, :]
    actions[c] = actionbloctest[c]
    bloc_num[c] = blocnum1

    next_states1 = np.zeros(statesize)
    next_actions1 = -1
    done_flags1 = 1
    blocnum1 = blocnum1 + 1
    bloc_num_reward += 1
    reward1 = -beat[0] * (SOFA[c]) + R3[c]

    bloc_num_reward = 0
    next_states[c] = next_states1
    next_actions[c] = next_actions1
    rewards[c] = reward1
    done_flags[c] = done_flags1
    c = c + 1  # 从0开始
    bloc_num = bloc_num[:c, :]
    states = states[: c, :]
    next_states = next_states[: c, :]
    actions = actions[: c, :]
    next_actions = next_actions[: c, :]
    rewards = rewards[: c, :]
    done_flags = done_flags[: c, :]

    bloc_num = np.squeeze(bloc_num)
    actions = np.squeeze(actions)
    rewards = np.squeeze(rewards)
    done_flags = np.squeeze(done_flags)

    # numpy形式转化为tensor形式
    batch_size = states.shape[0]
    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.LongTensor(actions).to(device)
    next_action = torch.LongTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)
    SOFA = torch.FloatTensor(SOFA).to(device)
    #bloc = torch.FloatTensor(bloc_num).to(device)
    batchs = (state, next_state, action, next_action, reward, done, bloc_num, SOFA)

    rec_phys_q = []
    rec_agent_q = []
    rec_agent_q_pro = []
    rec_phys_a = []
    rec_agent_a = []
    rec_sur = []
    rec_reward_user = []
    batch_s = 128
    uids = np.unique(bloc_num)
    num_batch = uids.shape[0] // batch_s  # 分批次
    for batch_idx in range(num_batch + 1):
        batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
        batch_user = np.isin(bloc_num, batch_uids)
        state_user = state[batch_user, :]
        next_state_user = next_state[batch_user, :]
        action_user = action[batch_user]
        next_action_user = next_action[batch_user]
        reward_user = reward[batch_user]
        done_user = done[batch_user]
        sur_Y90 = Y90[batch_user]
        bloc_num_user = bloc_num[batch_user]
        SOFA_user = SOFA[batch_user]

        batch = (state_user, next_state_user, action_user, next_action_user, reward_user, done_user, bloc_num_user, SOFA_user)
        q_output, agent_actions, phys_actions, Q_value_pro = do_eval(model, batch)

        q_output_len = range(len(q_output))
        agent_q = q_output[q_output_len, agent_actions]
        phys_q = q_output[q_output_len, phys_actions]

        rec_agent_q.extend(agent_q.detach().numpy())
        rec_agent_q_pro.extend(Q_value_pro.detach().numpy())

        rec_phys_q.extend(phys_q.detach().numpy())
        rec_agent_a.extend(agent_actions.detach().numpy())
        rec_phys_a.extend(phys_actions.detach().numpy())
        rec_sur.extend(sur_Y90)
        rec_reward_user.extend(reward_user.detach().numpy())

    np.save(f'{save_path}/shencunlv.npy', rec_sur)
    np.save(f'{save_path}/agent_bQ.npy', rec_agent_q)
    np.save(f'{save_path}/phys_bQ.npy', rec_phys_q)
    np.save(f'{save_path}/reward.npy', rec_reward_user)

    np.save(f'{save_path}/agent_actionsb.npy', rec_agent_a)
    np.save(f'{save_path}/phys_actionsb.npy', rec_phys_a)

    np.save(f'{save_path}/rec_agent_q_pro.npy', rec_agent_q_pro)

def do_test_ac(save_path, model, Xtest, actionbloctest, bloctest, Y90, SOFA, reward_value, beat):
    bloc_max = max(bloctest)  # Max is 20 phases
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
    R3 = r2[:, 0]
    R4 = (R3 + reward_value) / (2 * reward_value)
    RNNstate = Xtest
    print('####  Generating test trajectory ####')
    statesize = int(RNNstate.shape[1])
    states = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), statesize))
    actions = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1), dtype=int)
    next_actions = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1), dtype=int)
    rewards = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    next_states = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), statesize))
    done_flags = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    bloc_num = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    blocnum1 = 1
    c = 0

    bloc_num_reward = 0
    for i in range(RNNstate.shape[0] - 1):  # Loop through each row
        states[c] = RNNstate[i, :]
        actions[c] = actionbloctest[i]
        bloc_num[c] = blocnum1
        if (bloctest[i + 1] == 1):  # End of trace for this patient
            next_states1 = np.zeros(statesize)
            next_actions1 = -1
            done_flags1 = 1
            blocnum1 = blocnum1 + 1
            bloc_num_reward += 1
            reward1 = -beat[0] * (SOFA[i]) + R3[i]
            bloc_num_reward = 0
        else:
            next_states1 = RNNstate[i + 1, :]
            next_actions1 = actionbloctest[i + 1]
            done_flags1 = 0
            blocnum1 = blocnum1
            reward1 = -beat[1] * (SOFA[i + 1] - SOFA[i])
            bloc_num_reward += 1
        next_states[c] = next_states1
        next_actions[c] = next_actions1
        rewards[c] = reward1
        done_flags[c] = done_flags1
        c = c + 1
    states[c] = RNNstate[c, :]
    actions[c] = actionbloctest[c]
    bloc_num[c] = blocnum1

    next_states1 = np.zeros(statesize)
    next_actions1 = -1
    done_flags1 = 1
    blocnum1 = blocnum1 + 1
    bloc_num_reward += 1
    reward1 = -beat[0] * (SOFA[c]) + R3[c]
    bloc_num_reward = 0
    next_states[c] = next_states1
    next_actions[c] = next_actions1
    rewards[c] = reward1
    done_flags[c] = done_flags1
    c = c + 1
    bloc_num = bloc_num[:c, :]
    states = states[: c, :]
    next_states = next_states[: c, :]
    actions = actions[: c, :]
    next_actions = next_actions[: c, :]
    rewards = rewards[: c, :]
    done_flags = done_flags[: c, :]

    bloc_num = np.squeeze(bloc_num)
    actions = np.squeeze(actions)
    rewards = np.squeeze(rewards)
    done_flags = np.squeeze(done_flags)

    # Convert numpy arrays to tensors
    batch_size = states.shape[0]
    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.LongTensor(actions).to(device)
    next_action = torch.LongTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)
    batchs = (state, next_state, action, next_action, reward, done, bloc_num)

    rec_phys_q = []
    rec_agent_q = []
    rec_agent_q_pro = []
    rec_phys_a = []
    rec_agent_a = []
    rec_sur = []
    rec_reward_user = []
    batch_s = 128
    uids = np.unique(bloc_num)
    num_batch = uids.shape[0] // batch_s  # Split into batches
    for batch_idx in range(num_batch + 1):
        batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
        batch_user = np.isin(bloc_num, batch_uids)
        state_user = state[batch_user, :]
        next_state_user = next_state[batch_user, :]
        action_user = action[batch_user]
        next_action_user = next_action[batch_user]
        reward_user = reward[batch_user]
        done_user = done[batch_user]
        sur_Y90 = Y90[batch_user]

        batch = (state_user, next_state_user, action_user, next_action_user, reward_user, done_user)
        # Perform evaluation using Actor-Critic model
        q_output, agent_actions, phys_actions, Q_value_pro = do_eval(model, batch)

        # Extract Q-values and actions for analysis
        q_output_len = range(len(q_output))
        agent_q = q_output[q_output_len, agent_actions]
        phys_q = q_output[q_output_len, phys_actions]

        rec_agent_q.extend(agent_q.detach().numpy())
        rec_agent_q_pro.extend(Q_value_pro.detach().numpy())
        rec_phys_q.extend(phys_q.detach().numpy())
        rec_agent_a.extend(agent_actions.detach().numpy())
        rec_phys_a.extend(phys_actions.detach().numpy())
        rec_sur.extend(sur_Y90)
        rec_reward_user.extend(reward_user.detach().numpy())

    # Save results to specified path
    np.save(f'{save_path}/shencunlv.npy', rec_sur)
    np.save(f'{save_path}/agent_bQ.npy', rec_agent_q)
    np.save(f'{save_path}/phys_bQ.npy', rec_phys_q)
    np.save(f'{save_path}/reward.npy', rec_reward_user)

    np.save(f'{save_path}/agent_actionsb.npy', rec_agent_a)
    np.save(f'{save_path}/phys_actionsb.npy', rec_phys_a)

    np.save(f'{save_path}/rec_agent_q_pro.npy', rec_agent_q_pro)
