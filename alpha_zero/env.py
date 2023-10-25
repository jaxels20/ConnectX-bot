from kaggle_environments import evaluate, make, utils


def create_env():
    env = make("connectx", debug=True)
    env.render()
    return env


