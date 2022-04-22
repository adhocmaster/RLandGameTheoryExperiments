# Plan for different models

## Reward functions
1. The base model is too jumpy. We can add a negative reward for jumps. They should plan well. 
2. Added reward for scores
3. Monte Carlo style training on accumulated rewards after an episode ends.
5. Target net update: weight based on episode reward? or if some episode has high reward, it should have better probability of updating the target weight. Currently periodical update does not make a lot of sense.

## Frame skips
1. Currently action is repeatedly executed in skipped frames. Is it necessary? 
2. Framek skips are too low. 

## Why Marios is stuck sometimes (right only actions):
1. As Mario cannot see far enough, it may make a jump and stuck. To avoid this:
    i. make state longer (or more frame skips so that it can always see where it's gonna land and a little further.). But skipping more frame means that we may lost valuable information. Skipping 4 frames and stacking 5 states essentially means that we have more than 1 second of data if the fps is 15.
    ii. Allow left move action, but this will increase the training time greatly.
    iii. Change the reward function:
        a. The c part is probably not capturing much difference in time. Maybe reward needs to be more dynamic. Interestingly, the c-reward depends on the time difference, which does not change in 15 frames! So, we are losing a lot of computation power.
        b. add reward for reaching a flag.
        c. increase memory
