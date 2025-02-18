import numpy
import random
import chess
import conv
import chessenv
import chessenvPGN
import os
from pathlib import Path
import tensorflow.keras.optimizers as optimizers
import tensorflow
tensorflow.get_logger().setLevel('ERROR')
import movegeneration



class DeepQLearning:
    def __init__(self, env:chessenv, inputShape, memorySize, batchSize, gamma, epsilon, epsilonMin, epsilonDecay, training = True, normalize = False):
        # Init vals
        
        self.convNet = conv.convNet(inputShape, 16, 3, norm = normalize)
        self.model = self.convNet.model
        self.targetNet = conv.convNet(inputShape, 16, 3, norm = normalize)
        self.targetModel = self.targetNet.model
        self.targetModel.set_weights(self.model.get_weights())
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.env = env
        self.startEpisode = 0
        self.memorySize = memorySize
        self.batchSize = batchSize
        self.model.summary()
        self.targetModel.summary()
        self.modelName = "3_model"
        self.targetModelName = "3_targetModel" # 3_model and 3_targetModel are for irl data
        self.training = training
        

    def remember(self, state, action_idx, reward, nextState, done, turn):
        self.memory.append((state, action_idx, reward, nextState, done, turn))
        if len(self.memory) > self.memorySize:
            self.memory.pop(0)

    def random_move(self, _random = False):
        if _random:
            legalMoves = self.env.get_board().legal_moves
            legalMoves = list(legalMoves)
        else:
            legalMoves = [movegeneration.next_move(2, self.env.get_board())]
        random_move_array, idx = self.env.encode_move(random.choice(legalMoves), False, self.env.get_board().turn)
        return random_move_array, idx

    def act(self, state):
        if numpy.random.rand() <= self.epsilon:
            #get legal moves
            return self.random_move(True)
        
        # filter legal moves
        legalMoves = self.env.board.legal_moves
        legalMoves = list(legalMoves)
        legalMoves = [self.env.encode_move(move, True, self.env.board.turn)[1] for move in legalMoves]
        actValues = self.model.predict(state, verbose=1 if self.training else 0)[0]
        actValues = [actValues[move] for move in legalMoves]
        mx = legalMoves[numpy.argmax(actValues) if self.env.board.turn else numpy.argmin(actValues)]
        arr = numpy.zeros(shape=[76, 8, 8])
        arr[mx[0]][mx[1]][mx[2]] = 1
        return arr, (mx[0], mx[1], mx[2])
    
    
    def replay(self):
        if len(self.memory) < self.batchSize:
            return
        samples = random.sample(self.memory, self.batchSize)
        for sample in samples:
            state, _, reward, nextState, done, _ = sample
            # target = self.model.predict(state, verbose=1 if self.training else 0)
            if done:
                target:numpy.ndarray = numpy.array([[reward]])
            else:
                Q_future = self.targetModel.predict(nextState, verbose=1 if self.training else 0)
                target = [[reward]] + Q_future * self.gamma
            try:
                self.model.fit(state, target, epochs=1, verbose=1 if self.training else 0)
            except Exception as e:
                print(state.shape)
                print(target)
                raise e
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def evaluate_board(self, board):
        state = numpy.reshape(chessenv.get_bitboard(board), [1, 12, 8, 8])
        v = self.model.predict(state, verbose=1 if self.training else 0)[0][0]
        # if on linux use os.system('clear')
        if self.training:
            if os.name == 'nt':
                os.system('cls')
            elif os.name == 'posix':
                os.system('clear')
        return v

    def load(self, model_name, target_model_name, start_episode = 0):
        self.model.load_weights(model_name)
        self.targetModel.load_weights(target_model_name)
        # start episode from where we left off, only used for training
        self.startEpisode = start_episode

    def save(self, model_name, target_model_name, episode = 0):
        self.model.save(model_name+str(episode)+".h5")
        self.targetModel.save(target_model_name+str(episode)+".h5")

    def train(self, episodes):
        assert self.startEpisode < episodes
        self.epsilon = max(self.epsilonMin, self.epsilon * (self.epsilonDecay ** self.startEpisode))
        print(self.startEpisode, self.epsilon)
        for episode in range(episodes):
            episode += self.startEpisode
            # skip episodes if we are loading from a checkpoint
            print("Episode: ", episode)
            state = self.env.reset()
            state = numpy.reshape(state, [1, 12, 8, 8])
            done = False
            valid = True
            while (not done) and valid:
                # nextMove, reward = movegeneration.minimax_root_with_value(1, self.env.board, True, self, self.epsilon, episode)
                nextMove, reward = self.env.next_move(1, self.env.board, True, self, self.epsilon, episode)
                if nextMove is not None:
                    try:
                        self.env.board.push(nextMove)
                    except:
                        nextMove = None
                if self.env.board.is_checkmate():
                    print(self.env.board)
                    print("\n\n\nCheckmate for ", "white\n\n\n" if not self.env.board.turn else "black\n\n\n")
                if self.env.board.is_stalemate():
                    print(self.env.board)
                    print("\n\n\nStalemate\n\n\n")
                if self.env.board.is_insufficient_material():
                    print(self.env.board)
                    print("\n\n\nInsufficient Material\n\n\n")
                if self.env.board.is_fivefold_repetition():
                    print(self.env.board)
                    print("\n\n\nFivefold Repetition\n\n\n")
                if self.env.board.status()!=chess.Status.VALID:
                    print("\n\n\nInvalid\n\n\n")
                done = self.env.board.is_game_over() or self.env.board.is_stalemate() or self.env.board.is_insufficient_material() or self.env.board.is_checkmate() or nextMove is None
                valid = self.env.board.status() == chess.Status.VALID
                nextState = chessenv.get_bitboard(self.env.board)
                nextState = numpy.reshape(nextState, [1, 12, 8, 8])
                turn = self.env.board.turn
                self.remember(state, None, reward, nextState, done, turn)
                state = nextState
            self.replay()
            print("Finished episode: ", episode)
            if episode % 10 == 0:
                self.targetModel.set_weights(self.model.get_weights())
            if (episode+1) % 100 == 0 or episode+1 == episodes:    
                self.save(self.modelName, self.targetModelName, episode+1)

if __name__ == '__main__':
    env = chessenv.chessEnv(chess.Board())
    # env = chessenvPGN.chessEnvPGN(chess.Board(),  str(Path("PGN","Abdusattorov.pgn"))) 3_model and 3_targetModel are for irl data
    # dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.5, .95, 0.2, 0.95, False) # test with 500 memory size and gamma = 0.5 without normalization layer 1_model and 1_targetModel, self play data
    # dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.25, .95, 0.2, 0.95, training=True, normalize=False) # test with 500 memory size and gamma = 0.25 without normalization layer 2_model and 2_targetModel, self play data
    # dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.25, .95, 0.2, 0.95, training=True, normalize=True) # test with 500 memory size and gamma = 0.25 with normalization layer 4_model and 4_targetModel, self play data
    dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.125, .95, 0.2, 0.95, training=True, normalize=True) # test with 500 memory size and gamma = 0.125 with normalization layer 5_model and 5_targetModel, self play data
    # if os.name == 'nt':
    #     dql.load("checkpoints\\2_model500.h5", "checkpoints\\2_targetModel500.h5", 500)
    # else:
    #     dql.load("checkpoints/2_model500.h5", "checkpoints/2_targetModel500.h5", 500)
    dql.train(1000) # train for 1000 episodes
    
