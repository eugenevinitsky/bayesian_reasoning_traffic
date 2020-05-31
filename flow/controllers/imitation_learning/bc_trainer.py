import time
from collections import OrderedDict
import pickle
import numpy as np
import tensorflow as tf
import gym
import os

# params for saving rollout videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40

class BC_Trainer(object):

    def __init__(self, params):

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.sess = create_tf_session(self.params['use_gpu'], which_gpu=self.params['which_gpu'])

        # Set random seeds
        seed = self.params['seed']
        tf.set_random_seed(seed)
        np.random.seed(seed)

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]

        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.sess, self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        """
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n####### Iteration %i #######"%itr)

        # collect trajectories, to be used for training
        training_returns = self.collect_training_trajectories(itr,
                            initial_expertdata, collect_policy,
                            self.params['batch_size']) ## TODO implement this function below

        paths, envsteps_this_batch, train_video_paths = training_returns
        self.total_envsteps += envsteps_this_batch

        # relabel the collected obs with actions from a provided expert policy
        if relabel_with_expert and itr >= start_relabel_with_expert:
            paths = self.do_relabel_with_expert(expert_policy, paths) # TODO implement this function below
            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            self.train_agent() ## TODO implement this function below

            # save policy
            print('\nSaving agent\'s actor...')
            self.agent.actor.save(self.params['logdir'] + '/policy_itr_'+str(itr))

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.replay_buffer.sample_random_data(self.agent.batch_size)
