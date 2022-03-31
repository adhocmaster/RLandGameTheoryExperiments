# Plan for different models

## Reward functions
1. The base model is too jumpy. We can add a negative reward for jumps. They should plan well. 
2. Added reward for scores
3. Monte Carlo style training on accumulated rewards after an episode ends.
5. Target net update: weight based on episode reward? or if some episode has high reward, it should have better probability of updating the target weight. Currently periodical update does not make a lot of sense.

## Frame skips
1. Currently action is repeatedly executed in skipped frames. Is it necessary? 
2. 