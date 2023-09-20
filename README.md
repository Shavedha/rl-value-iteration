# EXPERIMENT 4: VALUE ITERATION ALGORITHM

## AIM
To implement value iteration algorithm for the given MDP

## PROBLEM STATEMENT
The environment is the 4*4 frozen lake where there are 5 terminal states namely 4 Hole states and 1 Goal State.It is a Stochastic environment.
### State Space:
{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
### Action Space:
The are 4 possible actions in this environment,
{0 (Left), 1(Right), 2(Up), 3(Down)}
### Reward:
* Reaches Goal State - +1
* Otherwise - 0
### Transition Probability:
* 33.33% - Agent moves in the desired direction.
* 66.66% - Agent moves in Orthogonal direction.
## VALUE ITERATION ALGORITHM
1. Initialize the value function V with zeros for each state.
2. Repeat the following until the change in V for all states is smaller than a threshold **theta**:
a. Initialize a Q-value function Q with zeros for each state-action pair.<br>
b. For each state s and action a, compute the Q-value using the Bellman equation: <br>
Q[s][a] = Σ[prob * (reward + gamma * V[next_state])] for all possible transitions (prob, next_state, reward, done).<br>
c. Update V by taking the maximum Q-value for each state: V[s] = max(Q[s]).<br>
3. Define a policy pi that selects actions by maximizing the Q-values: pi(s) = argmax(Q[s]).<br>
4. Return the final value function V and the corresponding policy pi.<br>

## VALUE ITERATION FUNCTION
```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if(np.max(np.abs(V-np.max(Q,axis=1))))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s , a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```
## OUTPUT
### Optimal Policy
<img width="369" alt="image" src="https://github.com/Shavedha/rl-value-iteration/assets/93427376/bdfb1757-0e17-4b45-8c07-5c60c94e8416">

### Optimal Value Function
<img width="356" alt="image" src="https://github.com/Shavedha/rl-value-iteration/assets/93427376/6876616a-69ac-461d-9708-e1b7fd7abb9c">

### Success Probability
<img width="428" alt="image" src="https://github.com/Shavedha/rl-value-iteration/assets/93427376/4ccab47a-e010-44e9-a91d-b0629558f4bd">

## RESULT
Thus Value iteration is implemented.
