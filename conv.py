import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers

class convNet:
    def __init__(self, inputShape):
        self.inputShape = inputShape
        
    def buildConvNet(self, convSize, convDepth):
        # Define input layers
        board_3d = layers.Input(shape=self.inputShape)
        boardSize = 8*8
        
        # This creates the hidden layers in the convNet
        x = board_3d
        for _ in range(convDepth):
            x = layers.Conv2D(filters=convSize, kernel_size=3, padding='same', activation='relu', data_format="channels_first")(x)
        # The curr size of x is (?, convSize, 8, 8)
        x = layers.Flatten()(x)
        x = layers.Dense((boardSize*76), 'relu')(x) # An array of size convSize * boardSize
        # output layer of dimension shape 8*8*76
        x = layers.Reshape((8, 8, 76))(x)
        self.model = models.Model(inputs=board_3d, outputs=x)
        return self.model
    
if __name__ == '__main__':
    convNet = convNet((12,8,8))
    model = convNet.buildConvNet(32, 2)
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
    model.summary()
    # utils.plot_model(model, to_file='model.png', show_shapes=True)
    