import os
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.regularizers import l2


class convNet:
    def __init__(self, inputShape, convSize, convDepth):
        #@Param inputShape : the shape of thew observation -> transformed board state
        #@param convSize : the size of the filter (will be the outputy size)
        #@param convDepth : Number of Convolution layers in the model
        self.inputShape = inputShape
        self.buildConvNet(convSize, convDepth)
        
    def buildConvNet(self, convSize, convDepth, hp=None):
        #@param convSize : the size of the filter (will be the outputy size)
        #@param convDepth : Number of Convolution layers in the model
        #@param hp : Hyperparameters for the model
        # Define input layers
        board_3d = layers.Input(shape=self.inputShape)
        boardSize = 8*8
        
        # This creates the hidden layers in the convNet
        x = board_3d
        for _ in range(convDepth):
            x = layers.Conv2D(filters=convSize, kernel_size=3, padding='same', activation='relu', data_format="channels_last", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        # The curr size of x is (?, convSize, 8, 8)
        x = layers.Flatten()(x)
        for _ in range(3):
            x = layers.Dense(boardSize*6, "relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x) # An array of size convSize * boardSize
        x = layers.Dropout(rate=0.25)(x)
        x = layers.Dense(1)(x)
        
        
        self.model = models.Model(inputs=board_3d, outputs=x)
        learning_rate = 5e-4
        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=["accuracy"],
        )
        
    
    def fit(self, train_x, train_y, epochs=100):
        assert train_x.shape[0] == train_y.shape[0]
        self.model.fit(train_x, train_y, epochs=epochs)
        
    def forward(self, board):
        #@param board : same dims as inputShape -> representatio of board state
        print(board.shape)
        return self.model.predict(board)
    
    
if __name__ == '__main__':
    net = convNet((12,8,8), 32, 2)
    model = net.model
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    model.summary()
    
    