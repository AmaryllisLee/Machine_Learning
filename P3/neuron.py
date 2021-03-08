class Neuron:

    def __init__(self, list_weights, bias, constant_learning_rate = 0.1):
        self.weights = list_weights
        self.bias    = bias
        self.n = constant_learning_rate

        self.iter = 0
        self.sum_error = 0
        self.rmse = 0
        self.error = 0

    def activation(self,  product_weights):
        return 1/(1 + 2.71828**(-product_weights - self.bias)) # sigmoid formule , e = 2.71828


    def calculate_output(self, input_arr):
        
        w_som = 0
        for i in range(len(input_arr)):# iterate through the index inputs e.g [0,0]
            weight = self.weights[i] # get weight 
            x = input_arr[i] # get x

            w_som += (weight*x) # increment w_som with the multiplication of weighti and xi
        output = self.activation(w_som) # apply the step function to w_Som
        return output

    def error_output(self, input_arr, d):
        """Bereken de error van de output neuron
        De formule voor de error luidt als volgt:
        o' = a0 * (1-a0) * - (d -  a0)
        a0 : output van de neuron met de feedforward
        d : verwachte output
        """
        a_out = self.calculate_output(input_arr)
        self.error = a_out * (1 - a_out) * - (d - a_out)
    
    def error_hidden(self, input_arr):
        "Bereken de error van een hidden layer"
        a_hidden = self.calculate_output(input_arr)
        self.error = a_hidden * ( 1- a_hidden) * self.error_output()

    def gradient(self, input_arr):
        "Bereken de gradient van een gewicht tussen neuron i en output neuron j."
        output = self.calculate_output(input_arr)
        return  output * self.error

    def delta(self, input_arr): 
        output = self.calculate_output(input_arr)
        self.delta_weights = self.n *  output * self.gradient 
        self.delta_bias    = self.n * self.error

    # Bereken de gewenste aanpassingen aan alle gewichten van output neuron j en sla deze op. 
    # We voeren de veranderingen nog niet door, omdat we de aanpassingen in het netwerk in één keer (dus allemaal tegelijkertijd) willen doorvoeren.?

    def update(self, input_arr):
        "Bijwerken vban de weights en de bias "
        delta_w , delta_b = self.delta(input_arr)
        for wi in self.weights:
            wi = wi - delta_w
        
        self.bias = self.bias - delta_b
    
    def __str__(self):
        return ("Weights: {}" + "\n" + "Biase/Threshold {}").format(self.weights, self.bias)

class NeuronLayer:

    def __init__(self):
        self.n_neurons = [] # list of Neuron objects

    def get_output(self, input_arr):
        """Calculate the output for each perceptron in n_perceptron

        Returns:
            [int]: lsit of containing the outputs of each perceptron in 0 or 1
        """
        input_next_layer = []
        for p in self.n_neurons: # for each perceptron
            g = p.calculate_output(input_arr) # get the outpout 
            input_next_layer.append(g) # ad to list
        return input_next_layer
    
    def __str__(self):
        layer_size = len(self.n_neurons)
        return ( 'Size of layer is {} perceptrons').format(layer_size)


class NeuronNetwork:

    def __init__(self):
        self.n_layers = []

    def feed_forward(self, input_arr):
        for layer in self.n_layers: # for each layer
            input_next_layer = layer.get_output(input_arr) # calculate the outputs of each perceptron in layer
            input_arr = input_next_layer# set input_next_layer as input_arr to be used as input for the next layer
        return input_arr
    
    def set_error(self):
        for layer in self.n_layers:
            for neuron in layer.n_neurons:
                neuron.error = 0
    
    def MSE(self, input_arr):
        pass

        
    def train(self,training_examples):
        cost = None
        while cost < 0.1:
            for example in training_examples:
                self.feed_forward(training_examples)
                self.set_error()
                self.update()
            
            cost = self.MSE(training_examples) 
        

        

    
    

