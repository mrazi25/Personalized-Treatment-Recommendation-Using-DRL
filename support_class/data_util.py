import os
import numpy as np
import pandas as pd
import joblib
import copy
import torch.optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
device = "cpu"

def run_kmeans(ncl, max_iter, sampl, nclustering):
    """
    Runs multiple K-Means clustering and selects the best model based on inertia.
    
    Parameters:
        ncl (int): Number of clusters.
        max_iter (int): Maximum number of iterations for each K-Means run.
        sampl (array-like): Data to cluster.
        n_runs (int): Number of times K-Means should be executed (default: 10).
    
    Returns:
        best_kmeans (KMeans): KMeans model with the lowest inertia.
    """
    best_kmeans = None
    best_inertia = np.inf

    for i in range(nclustering):
        print(f'Running K-Means iteration {i+1}/{nclustering}...')
        kmeans = KMeans(
            n_clusters=ncl,
            max_iter=max_iter,
            n_init=1,  # Ensures each run starts fresh
            verbose=False
        )
        kmeans.fit(sampl)

        print(f'Run {i+1}: Inertia = {kmeans.inertia_}')
        
        # Select the model with the best inertia (lowest value)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans

    print(f'Best Inertia: {best_inertia} after {nclustering} runs')
    return best_kmeans


def patient_clustering(save_path, type, X, blocs, ptid, MIMICtable, ncl, nclustering, prop, actionbloc, Y90, n_patient, n_components, random_state):
    
    # State Clustering
    N = X.shape[0]
    sampl = X[np.where(np.floor(np.random.rand(N, 1) + prop))[0], :]
    #sampl = X[np.random.rand(N) < prop, :]

    if type == 'train':
        kmeans_model_state = run_kmeans(ncl, 1, sampl, nclustering)
        #kmeans_model_state.fit(sampl)
        joblib.dump(kmeans_model_state, f'{save_path}/state_kmeans_model.pkl')
    else:
        kmeans_model_state = joblib.load(f'{save_path}/state_kmeans_model.pkl')

    idx = kmeans_model_state.predict(X)

    # Define Next State
    idx_next = []
    for i in range(len(X) - 1):
        if blocs[i + 1] != 1:
            idx_next.append(idx[i + 1])
        else:
            idx_next.append(ncl + 1 if MIMICtable['died_in_hosp'].iloc[i] == 1 else ncl)

    idx_next.append(ncl)  # Handle last element
    idx_next = np.array(idx_next)
    
    if type == 'train':
        # Define QLDATA3
        r = np.array([100, -100]).reshape(1, -1)
        r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
        qldata = np.column_stack([blocs, idx, actionbloc, Y90, r2])

        qldata3 = []
        absorbing_states = [ncl + 1, ncl]

        for i in range(len(qldata) - 1):
            qldata3.append(qldata[i, :4])
            if qldata[i + 1, 0] == 1:
                qldata3.append([qldata[i, 0] + 1, absorbing_states[int(qldata[i, 3])], -1, qldata[i, 4]])

        qldata3 = np.array(qldata3)

        # Create Transition Matrix
        transthres = 5
        transition_matrix = np.zeros((ncl + 2, ncl + 2, 25))
        transition_counts = np.zeros((ncl + 2, 25))

        for i in range(len(qldata3) - 1):
            if qldata3[i + 1, 0] != 1:
                S0, S1, action = int(qldata3[i, 1]), int(qldata3[i + 1, 1]), int(qldata3[i, 2])
                transition_matrix[S0, S1, action] += 1
                transition_counts[S0, action] += 1

        transition_counts[transition_counts <= transthres] = 0

        for i in range(ncl + 2):
            for j in range(25):
                if transition_counts[i, j] > 0:
                    transition_matrix[i, :, j] /= transition_counts[i, j]

        transition_matrix[np.isnan(transition_matrix)] = 0
        transition_matrix[np.isinf(transition_matrix)] = 0
        # Compute transition probability
        transition_prob = np.transpose(transition_matrix, (0, 2, 1))
        joblib.dump(transition_prob, f'{save_path}/transition_prob.pkl')
    else:
        transition_prob = joblib.load(f'{save_path}/transition_prob.pkl')

    #transition_prob = np.transpose(transition_matrix, (0, 2, 1))
    nact=5*5
    SA = np.zeros((ncl,nact))

    for state in range(len(SA)):
        for act in range(len(SA[state])):
            SA[state][act] = transition_prob[state][act].argmax()

    scaler = MinMaxScaler()
    SA_norm = scaler.fit_transform(SA)

    # Dimensionality Reduction (PCA)
    unique_ptids = np.unique(ptid)
    x_pca_input = np.array([SA_norm.flatten() for _ in range(len(unique_ptids))])

    sparsity = (SA == 0).mean(axis=0)
    print("Sparsity by feature:", sparsity)
    pca = PCA(n_components=n_components, random_state=random_state)
    x_pca = pca.fit_transform(x_pca_input)
    # Cumulative variance
    print(pca.components_[0])  # Loadings for the first component
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Number of components to retain 95% variance: {n_components}")
    joblib.dump(pca, f'{save_path}/pca_model.pkl')
    
    #x_pca = pca.fit_transform(x_pca_input)
    # Patient Clustering (K-Means)
    if type == 'train':
        kmeans_model_patient = KMeans(n_clusters=n_patient, random_state=random_state)
        kmeans_model_patient.fit(x_pca)
        joblib.dump(kmeans_model_patient, f'{save_path}/patients_kmeans_model.pkl')
    else:
        if not os.path.exists(f'{save_path}/patients_kmeans_model.pkl'):
            raise FileNotFoundError("Train the patient clustering model first.")
        kmeans_model_patient = joblib.load(f'{save_path}/patients_kmeans_model.pkl')
        #kmeans_model_patient.fit(x_pca)

    # Create Dataframe for Clustering
    batch_cls = pd.DataFrame({
        'bloc': blocs,
        'icustayid': ptid,
        'State': idx,
        'Action': actionbloc,
        'Next State': idx_next
    })

    new_row = {'bloc': 1, 'State': 0, 'Action': 0, 'Next State':0} # Data dummy untuk keperluan looping stay_id pasien
    batch_cls = pd.concat([batch_cls, pd.DataFrame([new_row])], ignore_index=True)

    # Initialize Transition Tracking
    temp_Count = copy.deepcopy(SA_norm)
    counter = copy.deepcopy(SA)
    temp_step_cl = []

    for i in range(len(batch_cls) - 1):
        if batch_cls['bloc'].iloc[i] == 1:
            temp_Count = copy.deepcopy(SA_norm)
            counter = copy.deepcopy(SA)

        state, action, next_state = int(batch_cls['State'].iloc[i]), int(batch_cls['Action'].iloc[i]), int(batch_cls['Next State'].iloc[i])
    
        if batch_cls['bloc'].iloc[i + 1] == 1:
            # temp_testCount = np.nan_to_num(temp_testCount, nan=0)  # Ganti NaN dengan 0
            # x_new_pca = pca.transform(temp_testCount.flatten().reshape(1, -1))
            # temp_step_cl.append(kmeans_model_patient.predict(x_new_pca)[0])

            x_new = temp_Count.flatten()
            x_new_pca = pca.transform(np.array(x_new).reshape(1, -1)) # 3 object
            step_cluster = kmeans_model_patient.predict(x_new_pca)[0]
            temp_step_cl.append(step_cluster)

        counter[state, action] = next_state
        #temp_testCount[state, action] = counter[state, action] / counter[state, action].sum()
        
        scaler = MinMaxScaler()
        temp_Count = scaler.fit_transform(counter)
            
    cluster_index = 0 # Mapping setiap cluster pasiennya kedalam dataframe

    for i in range(len(batch_cls)-1):
      batch_cls.loc[i, 'Patient Cluster'] = temp_step_cl[cluster_index]
      if(batch_cls['bloc'][i+1] == 1):
        cluster_index += 1

    batch_cls = batch_cls.drop(batch_cls.index[-1]) # Hapus data dummy yang tadi ditambahin

    # state_distribution = batch_cls["Patient Cluster"].value_counts().sort_index()

    # import matplotlib.pyplot as plt
    # # Tampilkan hasil
    # print(state_distribution)
    # plt.figure(figsize=(12, 6))
    # state_distribution.plot(kind="bar", color="skyblue")
    # plt.xlabel("State")
    # plt.ylabel("Jumlah Data")
    # plt.title("Sebaran Data per Unique State")
    # plt.xticks(rotation=90)
    # plt.show()
    return batch_cls

def patient_clustering_test(save_path, type, X, blocs, ptid, MIMICtable, ncl, nclustering, prop, actionbloc, Y90, n_patient, n_components, random_state):
    
    # State Clustering
    #N = X.shape[0]
    #sampl = X[np.where(np.floor(np.random.rand(N, 1) + prop))[0], :]
    #sampl = X[np.random.rand(N) < prop, :]

    
    kmeans_model_state = joblib.load(f'{save_path}/state_kmeans_model.pkl')

    idx = kmeans_model_state.predict(X)

    # Define Next State
    idx_next = []
    for i in range(len(X) - 1):
        if blocs[i + 1] != 1:
            idx_next.append(idx[i + 1])
        else:
            idx_next.append(ncl + 1 if MIMICtable['died_in_hosp'].iloc[i] == 1 else ncl)

    idx_next.append(ncl)  # Handle last element
    idx_next = np.array(idx_next)
    
    
    transition_prob = joblib.load(f'{save_path}/transition_prob.pkl')

    #transition_prob = np.transpose(transition_matrix, (0, 2, 1))
    nact=5*5
    SA = np.zeros((ncl,nact))

    for state in range(len(SA)):
        for act in range(len(SA[state])):
            SA[state][act] = transition_prob[state][act].argmax()

    scaler = MinMaxScaler()
    SA_norm = scaler.fit_transform(SA)

    # Dimensionality Reduction (PCA)
    unique_ptids = np.unique(ptid)
    x_pca_input = np.array([SA_norm.flatten() for _ in range(len(unique_ptids))])

    pca = PCA(n_components=n_components, random_state=random_state)
    x_pca = pca.fit_transform(x_pca_input)
    #pca = joblib.load(f'{save_path}/pca_model.pkl')
        #x_pca = pca.transform(x_pca_input)
    #x_pca = pca.fit_transform(x_pca_input)
    # Patient Clustering (K-Means)
    
    kmeans_model_patient = joblib.load(f'{save_path}/patients_kmeans_model.pkl')
    #kmeans_model_patient.fit(x_pca)
    # Create Dataframe for Clustering
    batch_cls = pd.DataFrame({
        'bloc': blocs,
        'icustayid': ptid,
        'State': idx,
        'Action': actionbloc,
        'Next State': idx_next
    })

    new_row = {'bloc': 1, 'State': 0, 'Action': 0, 'Next State':0} # Data dummy untuk keperluan looping stay_id pasien
    batch_cls = pd.concat([batch_cls, pd.DataFrame([new_row])], ignore_index=True)

    # Initialize Transition Tracking
    temp_testCount = copy.deepcopy(SA_norm)
    counter = copy.deepcopy(SA)
    #temp_step_cl = []

    for i in range(len(batch_cls) - 1):
        if batch_cls['bloc'].iloc[i] == 1:
            temp_testCount = copy.deepcopy(SA_norm)
            counter = copy.deepcopy(SA)

        #state, action, next_state = int(batch_cls['State'].iloc[i]), int(batch_cls['Action'].iloc[i]), int(batch_cls['Next State'].iloc[i])
        #counter[state, action] = next_state
        #temp_testCount[state, action] = counter[state, action] / counter[state, action].sum()
        state, action, next_state = int(batch_cls['State'].iloc[i]), int(batch_cls['Action'].iloc[i]), int(batch_cls['Next State'].iloc[i])

        #if batch_cls['bloc'].iloc[i + 1] == 1:
        temp_testCount = np.nan_to_num(temp_testCount, nan=0)  # Ganti NaN dengan 0
        #print(f"temp_testCount: {temp_testCount}")
        x_new_pca = pca.transform(temp_testCount.flatten().reshape(1, -1))
        #print(f"x_new_pca: {x_new_pca}")
        batch_cls.loc[i, 'Patient Cluster'] = kmeans_model_patient.predict(x_new_pca)[0]
        counter[state, action] = next_state
        scaler = MinMaxScaler()
        temp_testCount = scaler.fit_transform(counter)
            
    #cluster_index = 0 # Mapping setiap cluster pasiennya kedalam dataframe

    # for i in range(len(batch_cls)-1):
      
    #   if(batch_cls['bloc'][i+1] == 1):
    #     cluster_index += 1

    batch_cls = batch_cls.drop(batch_cls.index[-1]) # Hapus data dummy yang tadi ditambahin

    return batch_cls

def trajectory_generator(reformat5, data_type, RNNstate, actionbloc, blocs, Y90, SOFA):
    reward_value = 24
    beat1 = 0
    beat2 = 0.6
    beta3 = 0.3
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)

    ######################## Define Reward Function ###############################
    print('####  DEFINE REWARD FUNCTION  ####')
    SOFA = reformat5[data_type, 57]  # ***
    R3 = r2[:, 0]
    #R4 = (R3 + reward_value) / (2 * reward_value)
    #c = 0
    #bloc_max = max(blocstrain)
    print('####  GENERATE TRAJECTORY  ####')
    c = 0
    statesize = int(RNNstate.shape[1])
    states = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), statesize))
    actions = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1), dtype=int)
    next_actions = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1), dtype=int)
    rewards = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1))
    next_states = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), statesize))
    done_flags = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1))
    bloc_num = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1))
    blocnum1 = 1

    bloc_num_reward = 0
    for i in range(RNNstate.shape[0] - 1):  # Loop through each line
        states[c] = RNNstate[i, :]
        actions[c] = actionbloc[i]
        bloc_num[c] = blocnum1
        if (blocs[i + 1] == 1):  # end of trace for this patient
            next_states1 = np.zeros(statesize)
            next_actions1 = -1
            done_flags1 = 1
            blocnum1 = blocnum1 + 1
            bloc_num_reward += 1
            reward1 = -beat1 * (SOFA[i]) + R3[i]
            bloc_num_reward = 0
        else:
            next_states1 = RNNstate[i + 1, :]
            next_actions1 = actionbloc[i + 1]
            done_flags1 = 0
            blocnum1 = blocnum1
            reward1 = - beat2 * (SOFA[i + 1] - SOFA[i])
            bloc_num_reward += 1
        next_states[c] = next_states1
        next_actions[c] = next_actions1
        rewards[c] = reward1
        done_flags[c] = done_flags1
        c = c + 1

    states[c] = RNNstate[c, :]
    actions[c] = actionbloc[c]
    bloc_num[c] = blocnum1

    next_states1 = np.zeros(statesize)
    next_actions1 = -1
    done_flags1 = 1
    blocnum1 = blocnum1 + 1
    bloc_num_reward += 1
    reward1 = -beat1 * (SOFA[c]) + R3[c]

    bloc_num_reward = 0
    next_states[c] = next_states1
    next_actions[c] = next_actions1
    rewards[c] = reward1
    done_flags[c] = done_flags1
    c = c + 1

    bloc_num[c] = blocnum1
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
    batch_size = states.shape[0]
    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.LongTensor(actions).to(device)
    next_action = torch.LongTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)
    SOFAS = torch.LongTensor(SOFA).to(device)
    bloc_num = torch.tensor(bloc_num).to(device)
    batchs = (state, next_state, action, next_action, reward, done, bloc_num, SOFAS)
    return batchs