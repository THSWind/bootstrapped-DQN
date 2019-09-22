"""
Loads the environment, the model, the explorers,
parses arguments, trains the model
"""

import tensorflow as tf
from utils.modules import *
from utils.utils import PreprocessImg
from agent.model import Estimator, Agent


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, help='game environment name')
    parser.add_argument('--lr', type=float, default=25e-5,
                        help='learning rate for RMSProp')
    parser.add_argument('--gam', type=float, default=0.99,
                        help='discount factor gamma')
    parser.add_argument('--mb', type=int, default=32, help='minibatch size')
    parser.add_argument('--stp', type=int, default=50e6,
                        help='total number of steps to run the env for')

    return parser.parse_args()


def make_env(env_name):
    env = gym.make(args.env)
    # figure preprocessing


if __name__ == '__main__':
    args = parse_args()

    tf.reset_default_graph()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    q_estimator = Estimator(scope="q")
    target_estimator = Estimator(scope="target_q")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model(sess=sess, env=env, q_estimator=q_estimator,
              target_estimator=target_estimator, num_episodes=args.stp,
              replay_memory_size=10000, replay_memory_init_size=1000,
              discount_factor=args.gam, batch_size=args.mb,
              target_est_update=500)
        env.close()
        sess.close()
