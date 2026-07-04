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

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0

        # aqui estou repetindo para cada iteração
        # calculando um novo counter de valores para cada estado, e no final da iteração,
        # atualizando o counter de valores do agente com os novos valores calculados.
        for i in range(iterations):
            new_values = util.Counter()
            for state in mdp.getStates():
                if mdp.isTerminal(state):
                    new_values[state] = 0
                else:
                    action_values = []
                    for action in mdp.getPossibleActions(state):
                        q_value = self.computeQValueFromValues(state, action)
                        action_values.append(q_value)
                    new_values[state] = max(action_values)
            self.values = new_values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # aqui para cada estado, estou calculando o valor Q para cada ação possível
        # somando a probabilidade de transição multiplicada pelo valor do próximo estado 
        # e o desconto aplicado ao valor do próximo estado.
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q_value += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # se for um estado terminal retornar none, caso contrário, calcular 
        # o valor Q para cada ação possível e retornar a ação com o maior valor Q.
        if self.mdp.isTerminal(state):
            return None
        else:
            q_values = {}
            for action in self.mdp.getPossibleActions(state):
                q_values[action] = self.computeQValueFromValues(state, action)
            best_action = max(q_values, key=q_values.get)
            return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
