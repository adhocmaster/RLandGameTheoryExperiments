from os import stat


class GymUtils:

    @staticmethod
    def run(env, steps=100):
        done = True
        for step in range(steps):
            if done:
                state = env.reset()
            state, reward, done, info = env.step(env.action_space.sample())
            env.render()

        
    @staticmethod
    def runAndClose(env, steps=100):
        GymUtils.run(env, steps)
        env.close()
    