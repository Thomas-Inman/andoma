import numpy
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks

class convNet:
    def __init__(self, inputShape, convSize, convDepth):
        #@Param inputShape : the shape of thew observation -> transformed board state
        #@param convSize : the size of the filter (will be the outputy size)
        #@param convDepth : Number of Convolution layers in the model
        self.inputShape = inputShape
        self.buildConvNet(convSize, convDepth)
        
    def buildConvNet(self, convSize, convDepth):
        # Define input layers
        board_3d = layers.Input(shape=self.inputShape)
        boardSize = 8*8
        
        # This creates the hidden layers in the convNet
        x = board_3d
        for _ in range(convDepth):
            x = layers.Conv2D(filters=convSize, kernel_size=3, padding='same', activation='relu', data_format="channels_last")(x)
        # The curr size of x is (?, convSize, 8, 8)
        x = layers.Flatten()(x)
        for _ in range(convDepth):
            x = layers.Dense( boardSize * 76, 'relu')(x) # An array of size convSize * boardSize
        x = layers.Reshape((76, 8, 8))(x)
        self.model = models.Model(inputs=board_3d, outputs=x)
        self.model.compile(optimizer = optimizers.Adam(5e-4), loss='mean_squared_error')
    
    def fit(self, train_x, train_y, epochs=100):
        assert train_x.shape[0] == train_y.shape[0]
        self.model.fit(train_x, train_y, epochs=epochs)
        
    def forward(self, board):
        #@param board : same dims as inputShape -> representatio of board state
        print(board.shape)
        return self.model.predict(board)
    

# net = convNet((8, 8, 14), 76, 3)
# print(net.model.summary())

# train_x = np.random.uniform(0, 1, size=(100, 8, 8, 14))
# train_y = np.random.uniform(0, 1, size=(100, 76*8*8))
# print("x: " + str(train_x.shape))
# print("Y: " + str(train_y.shape))
# net.fit(train_x, train_y, epochs=5)
# print(net.forward(np.random.uniform(1, 2, size=(8, 8, 14))))
    
if __name__ == '__main__':
    net = convNet((12,8,8), 32, 2)
    model = net.model
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
    model.summary()
    # utils.plot_model(model, to_file='model.png', show_shapes=True)
    