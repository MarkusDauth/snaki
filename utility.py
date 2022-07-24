def define_parameters():
    params = dict()

    # Training and test
    params['episodes'] = 1000 # episodes for training OR test
    params['train'] = False # false = training run, which does not change NN

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

    return params
