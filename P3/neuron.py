class Neuron:

    def __init__(self, list_weights, bias):
        self.weights = list_weights
        self.bias    = bias

        self.iter = 0
        self.sum_error = 0
        self.rmse = 0

    def activation(self,  product_weights):
        return 1/(1 + 2.71828**(-product_weights)) # sigmoid formule , e = 2.71828


    def calculate_output(self, input_arr):
        
        w_som = 0
        for i in range(len(input_arr)):# iterate through the index inputs e.g [0,0]
            weight = self.weights[i] # get weight 
            x = input_arr[i] # get x

            w_som += (weight*x) # increment w_som with the multiplication of weighti and xi
        output = self.activation(w_som) # apply the step function to w_Som
        return output
    
    
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
    

    

    
    

