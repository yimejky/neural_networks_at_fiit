"""
Copy your implementation of MultilayerPerceptron class from your notebook here for submission.
Also append training related commands, such as model.compile, model.fit, etc.
"""
class MultilayerPerceptron(keras.Model):  # Subclassing
    
    def __init__(self, dim_output, dim_hidden, num_layers=1, activation=keras.activations.linear):
        super(MultilayerPerceptron, self).__init__(name='multilayer_perceptron')
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        # Within Model.__init__ we initialize all the layers we will use
        self.hidden_layers = []
        for _ in range(num_layers):
            layer = keras.layers.Dense(units=dim_hidden, activation=activation)
            self.hidden_layers.append(layer)
        self.layer_o = keras.layers.Dense(units=dim_output, activation=keras.activations.softmax)

    def call(self, x):  # call defines the flow of the computation, e.g. in this particular model
                        # we simply call the two layers one after the oter
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        y = self.layer_o(h)
        return y

    def run_to_tensorboard(self):
        model = MultilayerPerceptron(
            dim_output=3,
            dim_hidden=32,
            num_layers=3,
            activation=keras.activations.sigmoid)

        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.003),
            loss='categorical_crossentropy', # 'mean_squared_error'
            metrics=['accuracy'])

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs", timestamp()),
            histogram_freq=1)

        model.fit(
            x=data.x,
            y=data.y,
            batch_size=4,
            epochs=20,
            validation_split=0.2,
            callbacks=[tensorboard_callback],  # Callback
            verbose=0)  # Supressing text output