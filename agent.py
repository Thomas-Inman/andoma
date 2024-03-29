import numpy
import random
import chess


class QLearning:
  def __init__(self, env, alpha, gamma=0, epsilon=0.05, temp=0, nb_layers=2, decay=False):
    # Init vals
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.QM = QMatrix(nb_layers,10,1,self.env.observation_space,self.env.action_space,self.env)
    self.Q = self.QM.Q
    self.epsilon = epsilon
    self.nep = 1
    self.decay = decay
  def select_action(self, s, greedy=False):
    s = self.QM.get_states(s)
    if greedy:
      # if finished training, then choose the optimal policy
      try:
        # print(list(numpy.argmax(self.QM.Q[tuple([index])+tuple(x)]) for index, x in enumerate(s)))
        ls_r = numpy.bincount(list(numpy.argmax(self.QM.Q[tuple([index])+tuple(x)]) for index, x in enumerate(s)))
        winner = numpy.argwhere(ls_r == numpy.amax(ls_r)).flatten().tolist()
        r = numpy.random.choice(winner)
        # print(r)
      except Exception as e:
        print(s)
        raise e
      return r
    else:
      eps_decayed = max(0, self.epsilon-self.nep*0.001) if self.decay else self.epsilon# decay rate
      try:
        ls_r = numpy.bincount(list(numpy.argmax(self.QM.Q[tuple([index])+tuple(x)]) for index, x in enumerate(s)))
        winner = numpy.argwhere(ls_r == numpy.amax(ls_r)).flatten().tolist()
        r = numpy.random.choice(winner)
      except Exception as e:
        print(s)
        raise e
      return numpy.random.choice([numpy.random.choice([random.randint(0,self.env.action_space.n-1)]),r], p=[eps_decayed,1-eps_decayed])#numpy.argmax(softmax(self.Q[s], self.temp))

  def update(self, s, a, r, s_prime, a_prime, done=False):
    # Select the best action
    if done:
      return
    s2 = self.QM.get_states(s)
    s2_prime = self.QM.get_states(s_prime)
    self.nep = self.nep+1
    for i1,ls in enumerate(s2):
      for i2,ls2 in enumerate(s2_prime):
        self.QM.Q[tuple([i1])+tuple(ls)][a] = self.QM.Q[tuple([i1])+tuple(ls)][a] + self.alpha*(r + (self.gamma*numpy.max(self.QM.Q[tuple([i2])+tuple(ls2)])) - self.QM.Q[tuple([i1])+tuple(ls)][a])
