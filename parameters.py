def init_parameters():
    params = dict()

    # Training and test
    params['method'] = 'ppo' # 'qlearning' | 'sarsa' | 'ppo' | 'euclidean'
    params['train'] = True #   false = test run, which does not change NN   
    params['show_gui'] = False # disabling the GUI improves training speed
    params['device'] = 'cuda' # 'cuda' if torch.cuda.is_available() else 'cpu'

    # game settings
    params['game_field_size'] = 6 # must be even and atleast 6
    params['early_episode_end_steps'] = 100 # Agent has X steps time to find an apple. This value is increased by X, if an apple is found.

    # Hyperparameter for all RL methods
    params['train_episodes'] = 1500 # episodes for training. should be atleast 1000
    params['learning_rate'] = 0.00013629 #original is 0.001
    params['gamma'] = 0.90 # discount factor

    # only PPO 
    params['n_latent_var'] = 8 # number of neurons in hidden layer in NN
    params['ppo_update_timestep'] = 50 # update the policy every x game steps
    params['betas'] = (0.9, 0.999) # beta factor
    params['k_epochs'] = 80 # update policy for K epochs
    params['eps_clip'] = 0.2 # clip parameter for PPO

    # only qlearning and sarsa
    params['epsilon_decay_linear'] = 1.0/params['train_episodes'] # only used in
    params['batch_size'] = 32 # original was 1000
    params['memory_size'] = 2000 # original was 2500

    # test settings
    params['test_episodes'] = 1000 # number of episodes for test run


    return params
