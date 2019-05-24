# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        states = self.mdp.getStates()
        for state in states:
            self.values[(state,0)] = 0
            self.values[state] = (None,0)

        for i in range(1,self.iterations+1):
            for state in states:
                a_list = []
                v_list = []
                for action in self.mdp.getPossibleActions(state):
                    value = 0 
                    t_states = self.mdp.getTransitionStatesAndProbs(state,action)
                    for tpair in t_states:
                        nextState,tprob = tpair
                        value = value + tprob*(self.mdp.getReward(state,action,nextState) + self.discount*self.values[(nextState,i-1)])
                    a_list.append(action)
                    v_list.append(value)
                if v_list == []:
                    maxv = 0
                    maxa = None
                else:
                    maxv = max(v_list)
                    a_index = v_list.index(maxv)
                    maxa = a_list[a_index]
                self.values[(state,i)] = maxv
                self.values[state] = (maxa,maxv)





    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state][1]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qvalue = 0
        tpairs = self.mdp.getTransitionStatesAndProbs(state,action)
        for tpair in tpairs:
            nextState,tprob = tpair
            qvalue += tprob*(self.mdp.getReward(state,action,nextState) + self.discount*self.values[nextState][1])
        return qvalue
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        return self.values[state][0]





        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        gamma = self.discount
        for state in states:
            self.values[state] = (None,0)

        for i in range(0,self.iterations):
            state = states[i%(len(states))]
            if self.mdp.isTerminal(state):
                continue
            a_list = []
            v_list = []
            for action in self.mdp.getPossibleActions(state):
                value = 0 
                t_states = self.mdp.getTransitionStatesAndProbs(state,action)
                for tpair in t_states:
                    nextState,tprob = tpair
                    reward = self.mdp.getReward(state,action,nextState)
                    value = value + tprob*(reward + gamma*self.values[nextState][1])
                a_list.append(action)
                v_list.append(value)
        
            if v_list == []:
                maxv = 0
                maxa = None
            else:
                maxv = max(v_list)
                a_index = v_list.index(maxv)
                maxa = a_list[a_index]
            self.values[state] = (maxa,maxv)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        gamma = self.discount
        for state in self.mdp.getStates():
            self.values[state] = (None,0)
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
        sweepQueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for pair in self.mdp.getTransitionStatesAndProbs(state, action):
                    if pair[1] > 0:
                        predecessors[pair[0]].add(state)
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            diff = abs(self.getValue(state) - max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]))
            sweepQueue.push(state, -diff)

        for i in range(0, self.iterations):
            if sweepQueue.isEmpty():
                break
            state = sweepQueue.pop()
            a_list = []
            v_list = []
            for action in self.mdp.getPossibleActions(state):
                value = 0 
                t_states = self.mdp.getTransitionStatesAndProbs(state,action)
                for tpair in t_states:
                    nextState,tprob = tpair
                    reward = self.mdp.getReward(state,action,nextState)
                    value = value + tprob*(reward + gamma*self.values[nextState][1])
                a_list.append(action)
                v_list.append(value)
    
            if v_list == []:
                maxv = 0
                maxa = None
            else:
                maxv = max(v_list)
                a_index = v_list.index(maxv)
                maxa = a_list[a_index]
            self.values[state] = (maxa,maxv)

            for pred in predecessors[state]:
                diff = abs(self.getValue(pred) - max([self.getQValue(pred, action) for action in self.mdp.getPossibleActions(pred)]))
                if diff > self.theta:
                    sweepQueue.update(pred, -diff)









