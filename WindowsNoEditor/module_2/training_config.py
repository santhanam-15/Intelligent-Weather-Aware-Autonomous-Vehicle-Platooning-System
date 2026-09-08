class TrainingConfig:
    def __init__(self):
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_clip = 0.2
        self.entropy_coefficient = 0.01
        self.value_loss_coefficient = 0.5
        self.rollout_length = 256
        self.minibatch_size = 64
        self.epochs = 10
        self.hidden_size = 64
        self.num_agents = 4
        self.decision_frequency = 15
        self.model_save_interval = 50

