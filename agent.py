import numpy
import random
import chess
import conv
import chessenv
import movegeneration
import tensorflow.keras.optimizers as optimizers



class DeepQLearning:
    def __init__(self, env:chessenv, inputShape, memorySize, gamma, epsilon, epsilonMin, epsilonDecay, batchSize=32):
        # Init vals
        self.convNet = conv.convNet(inputShape, 16, 5)
        self.model = self.convNet.model
        self.targetNet = conv.convNet(inputShape, 16, 5)
        self.targetModel = self.targetNet.model
        self.targetModel.set_weights(self.model.get_weights())
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.env = env
        self.memorySize = memorySize
        self.batchSize = batchSize
        self.model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
        self.model.summary()
        self.targetModel.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
        self.targetModel.summary()
        

    def remember(self, board, state, action_idx, reward, nextState, done, turn):
        self.memory.append((board, state, action_idx, reward, nextState, done, turn))
        if len(self.memory) > self.memorySize:
            self.memory.pop(0)

    def random_move(self):
        # legalMoves = self.env.get_board().legal_moves
        # legalMoves = list(legalMoves)
        legalMoves = [movegeneration.next_move(2, self.env.get_board())]
        random_move_array, idx = self.env.encode_move(random.choice(legalMoves), False, self.env.get_board().turn)
        return random_move_array, idx
    def act(self, state):
        if numpy.random.rand() <= self.epsilon:
            #get legal moves
            return self.random_move()
        
        # filter legal moves
        legalMoves = self.env.get_board().legal_moves
        legalMoves = list(legalMoves)
        legalMoves = [self.env.encode_move(move, True, self.env.get_board().turn)[1] for move in legalMoves]
        actValues = self.model.predict(state)[0]
        actValues = [actValues[move] for move in legalMoves]
        mx = legalMoves[numpy.argmax(actValues) if self.env.get_board().turn else numpy.argmin(actValues)]
        arr = numpy.zeros(shape=[76, 8, 8])
        arr[mx[0]][mx[1]][mx[2]] = 1
        # print(arr)
        return arr, (mx[0], mx[1], mx[2])
    
    
    def replay(self):
        if len(self.memory) < self.batchSize:
            return
        samples = random.sample(self.memory, self.batchSize)
        for sample in samples:
            sample_board, state, action, reward, nextState, done, turn = sample
            target = self.model.predict(state)
            if done:
                # print(target)
                # print(action)
                target[0][action] = reward
            else:
                # filter legal moves
                legalMoves = sample_board.legal_moves
                legalMoves = list(legalMoves)
                legalMoves = [self.env.encode_move(move, True, turn)[1] for move in legalMoves]
                Q_future = self.targetModel.predict(nextState)[0]
                Q_future = [Q_future[move] for move in legalMoves]
                try:
                    Q_future = numpy.max(Q_future) if turn else numpy.min(Q_future)
                except:
                    Q_future = 0

                target[0][action] = reward + Q_future * self.gamma
            # print("FITTING")
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def train(self, episodes):
        for episode in range(episodes):
            
            print("Episode: ", episode)
            state = self.env.reset()
            state = numpy.reshape(state, [1, 2, 12, 8, 8])
            done = False
            valid = True
            while (not done) and valid:
                action, action_idx = self.act(state)
                if not self.env.board.is_legal(self.env.decode_move(action, self.env.get_board().turn)):
                    print("DEBUG: Illegal move")
                    action, action_idx = self.random_move()
                
                nextState, reward, done, valid = self.env.step(self.env.decode_move(action, self.env.get_board().turn))
                nextState = numpy.reshape(nextState, [1, 2, 12, 8, 8])
                turn = self.env.board.turn
                self.remember(self.env.board.copy(stack=False), state, action_idx, reward if self.env.board.turn else -reward, nextState, done, turn)
                state = nextState
            if self.env.board.status() == chess.Status.VALID:
                self.replay()
            print("Finished episode: ", episode)

            if episode % 10 == 0:
                self.targetModel.set_weights(self.model.get_weights())
                
        self.model.save("model2.h5")
        self.targetModel.save("targetModel2.h5")
    
    def load(self):
        self.model.load_weights("model.h5")
        self.targetModel.load_weights("targetModel.h5")

if __name__ == '__main__':
    env = chessenv.chessEnv(chess.Board())
    dql = DeepQLearning(env, (2, 12, 8, 8), 500, 0.9, 0.9, 0.1, 0.9, 32)
    dql.train(1000)
    # dql.load()
    
