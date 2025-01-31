import torch
import numpy as np
import warnings
import os
import torch.nn.functional as F

warnings.filterwarnings("ignore")
device = 'cpu' 

def do_eval(model, batchs, batch_size=128):
    #(state, next_state, action, next_action, reward, done) = batchs
    states, next_states, actions, next_actions, rewards, done_flags, bloc_num, SOFAS = batchs
    Q_value = model.Q(states)
    agent_actions = torch.argmax(Q_value, dim=1)
    phy_actions = actions
    Q_value_pro1 = F.softmax(Q_value)
    Q_value_pro_ind = torch.argmax(Q_value_pro1, dim=1)
    Q_value_pro_ind1 = range(len(Q_value_pro_ind))
    Q_value_pro = Q_value_pro1[Q_value_pro_ind1, Q_value_pro_ind]
    return Q_value, agent_actions, phy_actions, Q_value_pro
# def do_eval(model, batchs):
#     phys_q_ret = []
#     actions_ret = []
#     agent_q_ret = []
#     actions_taken_ret = []
#     error_ret = 0
#     gamma = 0.999 # discount factor
#     #print(f"Batch shape and content: {len(b)}, {b}")
#     #print(f"Batch content: {batchs}, length: {len(batchs)}")
#     states, next_states, actions, next_actions, rewards, done_flags, bloc_num, SOFAS = batchs
#     # Firstly, get the chosen actions at the next timestep using mainQN.predict()
#     actions_from_q1 = model.get_action(next_states)

#     # Q values for the next timestep from target network, as part of the Double DQN update
#     Q2 = model.get_action(next_states)

#     # Handle the case when a trajectory is finished
#     end_multiplier = 1 #- done_flags

#     # Target Q value using Q values from target, and actions from main
#     double_q_value = Q2[actions_from_q1]
#     # Definition of target Q
#     targetQ = rewards + (gamma * double_q_value * end_multiplier)

#     # Get the output q's, actions, and loss
#     q_output, actions_taken, abs_error = model.evaluate(states, targetQ, actions)
    
#     # Return the relevant q values and actions
#     phys_q = q_output[actions]  # Single index for 1D tensor
#     agent_q = q_output[actions_taken]
#     error = abs_error.mean()

#     # Update the return values
#     phys_q_ret.extend(phys_q)
#     actions_ret.extend(actions)
#     agent_q_ret.extend(agent_q)
#     actions_taken_ret.extend(actions_taken)
#     error_ret += error

#     Q_value = model.Q(states).to(device)
#     Q_value_pro1 = F.softmax(Q_value)
#     Q_value_pro_ind = torch.argmax(Q_value_pro1, dim=1)
#     Q_value_pro_ind1 = range(len(Q_value_pro_ind))
#     Q_value_pro = Q_value_pro1[Q_value_pro_ind1, Q_value_pro_ind]

#     return Q_value, phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, Q_value_pro, error_ret
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


def do_test(model_name, mimic_type, model, batchS, Y90, n_patient, ncl, cluster):
    state, next_state, action, next_action, reward, done, bloc_num, SOFA, clusters = batchS

    rec_phys_q = []
    rec_agent_q = []
    rec_agent_q_pro = []
    rec_phys_a = []
    rec_agent_a = []
    rec_sur = []
    rec_reward_user = []
    batch_s = 128
    error_ret = 0
    gamma = 0.999  # Discount factor
    uids = np.unique(bloc_num)

    # Correctly handle the final batch size to avoid out-of-bounds error
    num_batch = (uids.shape[0] + batch_s - 1) // batch_s
    
    for batch_idx in range(num_batch):
        batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
        batch_user = np.isin(bloc_num, batch_uids)
        state_user = state[batch_user, :]
        next_state_user = next_state[batch_user, :]
        action_user = action[batch_user]
        next_action_user = next_action[batch_user]
        reward_user = reward[batch_user]
        done_user = done[batch_user]
        sur_Y90 = Y90[batch_user]
        
        clusters = np.array(clusters)
        cluster_user = clusters[batch_user]
        cluster_num = int(max(clusters)) + 1
        
        # Print debugging information
        print(f"Cluster user: {cluster_user}")
        
        batch = (state_user, next_state_user, action_user, next_action_user, reward_user, done_user, bloc_num, SOFA, sur_Y90, cluster_user)
        
        for i in range(len(batch[0])):
            # Retrieve actions from the Q1 model
            actions_from_q1 = model[int(batch[9][i])].get_action(batch[1][i].unsqueeze(0).to(device))

            # Q values for the next timestep (Double DQN update)
            Q2 = model[int(batch[9][i])].get_action(batch[1][i].unsqueeze(0).to(device))

            # Handle the case when a trajectory is finished
            if Q2.dim() == 1 and Q2.shape[0] == 1:
                Q2 = Q2.unsqueeze(0)  # Ensure batch dimension

            # Prevent out-of-bounds indexing
            actions_from_q1 = torch.clamp(actions_from_q1, min=0, max=Q2.shape[0] - 1)

            # Index safely for double Q value
            double_q_value = Q2[actions_from_q1]
            end_multiplier = 1

            # Target Q value using Q values from target, and actions from main
            targetQ = batch[4][i] + (gamma * double_q_value * end_multiplier)
            
            with torch.no_grad():
                # Compute Q-values for the current states
                q_output_all = model[int(batch[9][i])].Q(batch[0][i].unsqueeze(0).to(device))

                # Get the Q-values for the specific actions taken
                batch_2_i = torch.tensor([batch[2][i]], dtype=torch.long, device=device)
                range_batch = torch.arange(q_output_all.shape[0]).long().to(device)

                q_output = q_output_all[range_batch, batch_2_i]

                # Predict the actions the agent would take
                actions_taken = torch.argmax(q_output_all, dim=1)
                abs_error = torch.abs(targetQ - q_output)

            # Ensure batch size is correct and update return values
            batch_2_i = torch.clamp(batch[2][i], min=0, max=q_output.shape[0] - 1)
            actions_taken = torch.clamp(actions_taken, min=0, max=q_output.shape[0] - 1)
            phys_q = q_output[batch_2_i]  # Single index for 1D tensor
            agent_q = q_output[actions_taken]
            error_ret += abs_error.mean()
            agent_actions = actions_taken
            phys_actions = batch[2][i]
            
            Q_value = model[int(batch[9][i])].Q(batch[0][i].unsqueeze(0).to(device)).to(device)
            Q_value_pro1 = F.softmax(Q_value)
            Q_value_pro_ind = torch.argmax(Q_value_pro1, dim=1)
            Q_value_pro_ind1 = range(len(Q_value_pro_ind))
            q_value_pro = Q_value_pro1[Q_value_pro_ind1, Q_value_pro_ind]

            # Record the relevant q values and actions
            # Replace extend() with np.append() for numpy arrays
            # rec_agent_q = np.append(rec_agent_q, agent_q.detach().numpy())
            # rec_agent_q_pro = np.append(rec_agent_q_pro, q_value_pro.detach().numpy())
            # rec_phys_q = np.append(rec_phys_q, phys_q.item())
            # rec_agent_a = np.append(rec_agent_a, agent_actions.detach().numpy())
            # rec_phys_a = np.append(rec_phys_a, phys_actions.item())
            # rec_sur = np.append(rec_sur, batch[8][i])
            # rec_reward_user = np.append(rec_reward_user, batch[4][i])
            rec_agent_q = np.concatenate((rec_agent_q, agent_q.detach().numpy()))
            rec_agent_q_pro = np.concatenate((rec_agent_q_pro, q_value_pro.detach().numpy()))
            rec_phys_q = np.concatenate((rec_phys_q, [phys_q.item()]))  # Wrap scalar in a list or array
            rec_agent_a = np.concatenate((rec_agent_a, agent_actions.detach().numpy()))
            rec_phys_a = np.concatenate((rec_phys_a, [phys_actions.item()]))  # Wrap scalar in a list or array
            rec_sur = np.concatenate((rec_sur, [batch[8][i]]))  # Wrap scalar in a list or array
            rec_reward_user = np.concatenate((rec_reward_user, [batch[4][i]]))  # Wrap scalar in a list or array

    
        # Ensure alignment of recorded values and prevent out-of-bounds error
        rec_sur = np.array(rec_sur)
        rec_agent_q = np.array(rec_agent_q)
        rec_phys_q = np.array(rec_phys_q)

        # Ensure we only use the subset of data for sorting
        # current_bloc_num = bloc_num[batch_user]
        
        print(f"rec_sur length: {len(rec_sur)}, rec_agent_q length: {len(rec_agent_q)}, rec_phys_q length: {len(rec_phys_q)}")
        
        # Sort by current bloc_num subset (not the full bloc_num)
        # sorted_indices = np.argsort(current_bloc_num)[:rec_sur.shape[0]]  # Slice to match length
        # rec_sur = rec_sur[sorted_indices]
        # rec_agent_q = rec_agent_q[sorted_indices]
        # rec_phys_q = rec_phys_q[sorted_indices]
        
    save_path = f'../../result/mimic{mimic_type}/{n_patient}_cluster/{ncl}_states/{model_name}'
    os.makedirs(save_path, exist_ok=True)
    
    # Save the results to the specified directory
    np.save(f'{save_path}/shencunlv_cls{n_patient}.npy', rec_sur)
    np.save(f'{save_path}/agent_bQ_cls{n_patient}.npy', rec_agent_q)
    np.save(f'{save_path}/phys_bQ_cls{n_patient}.npy', rec_phys_q)
    np.save(f'{save_path}/reward_cls{n_patient}.npy', rec_reward_user)
    np.save(f'{save_path}/agent_actionsb_cls{n_patient}.npy', rec_agent_a)
    np.save(f'{save_path}/phys_actionsb_cls{n_patient}.npy', rec_phys_a)
    np.save(f'{save_path}/rec_agent_q_pro_cls{n_patient}.npy', rec_agent_q_pro)

def do_test_ac(mimic_type, actor_critic_model, batchS, Y90, n_patient, ncl, cluster):
    state, next_state, action, next_action, reward, done, bloc_num, SOFA, clusters = batchS

    rec_phys_q = []
    rec_agent_q = []
    rec_agent_q_pro = []
    rec_phys_a = []
    rec_agent_a = []
    rec_sur = []
    rec_reward_user = []
    batch_s = 128
    gamma = 0.999  # Discount factor
    uids = np.unique(bloc_num)

    # Correctly handle the final batch size to avoid out-of-bounds error
    num_batch = (uids.shape[0] + batch_s - 1) // batch_s

    for batch_idx in range(num_batch):
        batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
        batch_user = np.isin(bloc_num, batch_uids)
        state_user = state[batch_user, :]
        next_state_user = next_state[batch_user, :]
        action_user = action[batch_user]
        next_action_user = next_action[batch_user]
        reward_user = reward[batch_user]
        done_user = done[batch_user]
        sur_Y90 = Y90[batch_user]

        clusters = np.array(clusters)
        cluster_user = clusters[batch_user]
        
        batch = (
            state_user,
            next_state_user,
            action_user,
            next_action_user,
            reward_user,
            done_user,
            bloc_num,
            SOFA,
            sur_Y90,
            cluster_user,
        )

        for i in range(len(batch[0])):
            # Get actions and Q-values from the Actor-Critic model
            state_tensor = batch[0][i].unsqueeze(0).to(device)
            next_state_tensor = batch[1][i].unsqueeze(0).to(device)

            # Predict policy and value for current and next state
            policy, value = actor_critic_model(state_tensor)
            next_policy, next_value = actor_critic_model(next_state_tensor)

            # Sample action and compute probabilities
            dist = torch.distributions.Categorical(policy)
            actions_taken = dist.sample()
            log_prob = dist.log_prob(actions_taken)

            # Compute Q-value for the action taken
            q_value = value[0]  # Critic output as the state value
            advantage = reward_user[i] + (gamma * next_value[0] * (1 - done_user[i])) - q_value

            # Record values for this sample
            rec_agent_q = np.append(rec_agent_q, q_value.item())
            rec_agent_q_pro = np.append(rec_agent_q_pro, log_prob.exp().item())
            rec_phys_q = np.append(rec_phys_q, value[action_user[i]].item())
            rec_agent_a = np.append(rec_agent_a, actions_taken.item())
            rec_phys_a = np.append(rec_phys_a, action_user[i])
            rec_sur = np.append(rec_sur, batch[8][i])
            rec_reward_user = np.append(rec_reward_user, reward_user[i])

    save_path = f'../../result/mimic{mimic_type}/{n_patient}_cluster/{ncl}_states/ActorCritic'
    os.makedirs(save_path, exist_ok=True)

    # Save the results to the specified directory
    np.save(f'{save_path}/shencunlv_cls{n_patient}.npy', rec_sur)
    np.save(f'{save_path}/agent_bQ_cls{n_patient}.npy', rec_agent_q)
    np.save(f'{save_path}/phys_bQ_cls{n_patient}.npy', rec_phys_q)
    np.save(f'{save_path}/reward_cls{n_patient}.npy', rec_reward_user)
    np.save(f'{save_path}/agent_actionsb_cls{n_patient}.npy', rec_agent_a)
    np.save(f'{save_path}/phys_actionsb_cls{n_patient}.npy', rec_phys_a)
    np.save(f'{save_path}/rec_agent_q_pro_cls{n_patient}.npy', rec_agent_q_pro)

