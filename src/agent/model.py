"""
Agent file which contains the estimator of the value function
and the DQN model.

Hyperparameters
    discount_factor: 0.99
    learning_rate: 25e-5
    target_update_steps: 10e3
    steps: 50e6
    steps_evaluated: 1e6
    memory_size: 1e6 (tuples)
    memory_sample_steps: 4
    minibatch_size: 32
"""

from utils.modules import *
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected

NUM_HEADS = 10
INPUT_SHAPE = [None, 84, 84, 4]


class Estimator():
    """
    Estimator class from the estimate of the value function
    Contains
        neuralnet: builds the NN to predict the value functions
        predict: computes the best action given the argmax value function
        switch_scope: switches the scope to estimate both Q and target Q
    """
    def __init__(self, scope='estimator'):
        self.scope = scope
        with tf.variable_scope(scope):
            self.X_place = tf.placeholder(shape=INPUT_SHAPE,
                                          dtype=tf.float32,
                                          name='X')
            self.y_place = tf.placeholder(shape=[None],
                                          dtype=tf.float32,
                                          name='y')
            self.actions_place = tf.placeholder(shape=[None],
                                                dtype=tf.int32,
                                                name='actions')

            self.neuralnet()
            # array of value function predictions (fc output)
            self.predictions_arr = []
            # array of the minimized loss from optimizer
            self.train_op_arr = []
            # array of the losses
            self.loss_arr = []

        def neuralnet(self, learning_rate=25e-5, discount_factor=0.99,
                      momentum=0.95):
            """
            Builds the simple neural network for Q estimation
            Input is an 84 x 84 x 4 tensor grayscale of the last four frames

            Architecture
                First conv layer: convolves input with 32 filters of size 8,
                stride 4
                Second conv layer: has 64 filters of size 4, stride 2
                Final conv layer: has 64 filters of size 3, stride 1
                FC layer: has 512 hidden units separated by ReLu
                FC layer: projects Q-values

                Optimizer: RMSProp with momentum 0.95
            """
            initializer = tf.contrib.layers.variance_scaling_initializer()
            output = conv2d(inputs=X_place, filters=32, kernel_size=(8, 8),
                            strides=4, padding='same',
                            kernel_initializer=initializer,
                            activation=tf.nn.relu)
            output = conv2d(inputs=output, filters=64, kernel_size=(4, 4),
                            strides=2, padding='same',
                            kernel_initializer=initializer,
                            activation=tf.nn.relu)
            output = conv2d(inputs=output, filters=64, kernel_size=(3, 3),
                            strides=1, padding='same',
                            kernel_initializer=initializer,
                            activation=tf.nn.relu)

            output = flatten(output)

            for _ in range(NUM_HEADS):
                fc = fully_connected(inputs=output,
                                     num_outputs=512,
                                     weights_initializer=initializer)
                fc = fully_connected(inputs=fc,
                                     num_outputs=N_OUTPUTS,
                                     weights_initializer=initializer,
                                     activation_fn=None)

                # append output of the fully connected layers to an array
                self.predictions_arr.append(fc)

                gather_indices = tf.range(tf.shape(self.X_place)[0]) * \
                    tf.shape(predictions_arr)[1] + self.actions_pl
                action_preds = tf.gather(tf.reshape(predictions_arr, [-1]),
                                         gather_indices)

                losses = tf.squared_distance(self.y_place, action_preds)
                loss = tf.reduce_mean(losses)

                optimizer = \
                    tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                              decay=discount_factor,
                                              momentum=momentum,
                                              name='RMSProp')
                glob_step = tf.contrib.framework.get_global_step()
                train_op = \
                    optimizer.minimize(loss,
                                       global_step=glob_step)

                self.loss_arr.append(loss)
                self.train_op_arr.append(train_op)

        def predict(self, sess, state, index):
            """
            Predicts the value function and chooses the best action from the
            prediction
            """
            q_val_pred = sess.run(self.predictions_arr[index],
                                  {self.X_place: np.expand_dims(state)})[0]
            best_action = np.argmax(q_val_pred)
            return best_action

        def optimize(self, sess, state, action, target, index):
            feed_dict = {self.X_place: state, self.y_place: target,
                         self.actions_pl: action}
            global_step, _, loss = sess.run(
                [tf.contrib.framework.get_global_step(),
                 self.train_op_arr[index], self.loss_arr[index]],
                feed_dict)
            return loss

        def switch_scope(sess, scope1, scope2):
            """
            Switches desired value function scope
            """
            s1_params = [t for t in tf.trainable_variables()
                         if t.name.startswith(scope1.scope)]
            s1_params = sorted(s1_params, key=lambda v: v.name)
            s2_params = [t for t in tf.trainable_variables()
                         if t.name.startswith(scope2.scope)]
            s2_params = sorted(s2_params, key=lambda v: v.name)

            update_ops = []
            for s1_v, s2_v in zip(s1_params, s2_params):
                op = s2_v.assign(s1_v)
                update_ops.append(op)

            sess.run(update_ops)


class Agent():
    """
    Agent class which runs the model and explorers
    Contains
        model: runs the bootstrapped DQN model
    """
    def __init__(self, config, env, sess, explorer):
        super(Agent, self).__init__(config)
        self.env = env

    def run_explorer(self):
        # to do
        pass

    def model(self, sess, env, q_estimator, target_estimator,
              num_episodes, replay_memory_size=10000,
              replay_memory_init_size=1000, discount_factor=0.99,
              batch_size=16, target_est_update=500):
        """
        The model function which trains the agent model
        Arguments
            sess: Current TensorFlow Session object
            q_estimator: value function estimation
            target_estimator: target value function estimation
            state_processor: Processed state frames
            num_episodes: Total number of episodes to be played
            replay_memory_size: Size of memory replay array
            replay_memory_init_size:
            discount_factor: Gamma discount factor
            batch_size: NN batch size
            target_est_update: How frequent to update target estimator
        """

        # Trajectory
        Transition = namedtuple("Transition", ["state", "action", "reward",
                                "next_state", "done"])
        replay_memory = []
        time_total = sess.run(tf.contrib.framework.get_global_step())

        VALID_ACTIONS = env.unwrapped.get_action_meanings()
        N_OUTPUTS = env.action_space.n

        # iterate through episodes
        for i_episode in tqdm(range(num_episodes)):

            # grab processed state from env
            state = env.reset()
            loss = None
            total_reward = 0
            # actions_tracker = [0, 1, 2, 3]
            index = np.random.randint(NUM_HEADS)

            # iterate through time
            for t in itertools.count():

                # checkpoint
                if time_total % target_est_update == 0:
                    q_estimator.switch_scope(sess, q_estimator,
                                             target_estimator)

                # estimates the action for the estimated Q
                action = q_estimator.predict(sess, state, index)
                actions_tracker.append(action)
                next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
                next_state = np.append(state[:, :, 1:],
                                       np.expand_dims(next_state, 2), axis=2)

                # if replay memory exceeds the allocated size
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)

                # appends current trajectory into the replay memory
                replay_memory.append(Transition(state, action, reward,
                                                next_state, done))
                total_reward += reward

                # if total time exceeds replay memory size
                if time_total > replay_memory_init_size:
                    for agent_train in range(NUM):
                        # for sampling random minibatch of transitions
                        samples = random.sample(replay_memory, batch_size)
                        states_batch, action_batch, reward_batch,
                        next_states_batch, done_batch = map(np.array,
                                                            zip(*samples))

                        # estimate next value functions and select best action
                        q_values_next = q_estimator.predict(sess,
                                                            next_states_batch,
                                                            agent_train)
                        best_actions = np.argmax(q_values_next, axis=1)

                        q_values_next_target = \
                            target_estimator.predict(sess, next_states_batch,
                                                     agent_train)

                        targets_batch = reward_batch + \
                            np.invert(done_batch).astype(np.float32) * \
                            discount_factor * \
                            q_values_next_target[np.arange(batch_size),
                                                 best_actions]

                        states_batch = np.array(states_batch)
                        loss = q_estimator.optimize(sess, states_batch,
                                                    action_batch,
                                                    targets_batch, agent_train)

                state = next_state
                time_total += 1

                if done:
                    print("Step {} ({}) @ Episode {}/{}, loss: {}".format(
                        t, time_total, i_episode + 1, num_episodes, loss),
                        end=", ")
                    print('reward %f, steps %d' % (total_reward, t), end=", ")
                    break
