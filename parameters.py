def init_parameters():
    params = dict()

    # Training and test
    params['method'] = 'sarsa' # 'qlearning' | 'sarsa' | 'ppo' | 'euclidean'
    params['train'] = True #   false = test run, which does not change NN   
    params['show_gui'] = False # disabling the GUI improves training speed

    # game settings
    params['game_field_size'] = 24 # must be even and atleast 6

    # Hyperparameter for all RL methods
    params['train_episodes'] = 200 # episodes for training. should be atleast 1000
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

    # not often changed
    params['test_episodes'] = 1000
    params['device'] = 'cuda' # 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    '''
    # Neural Network
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['epsilon_decay_linear'] = 1.0/params['episodes']
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['test'] = (params['train'] == False)
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    params['agent_type'] = 'q_learning' # 'q_learning' | 'sarsa' | 'expected_sarsa' | 'ppo'

    # Added by Markus
    params['display'] = False # Show GUI
    params['speed'] = 0 # Default = 50; fastest = 0
    params['bayesianopt'] = False
    '''

    return params
