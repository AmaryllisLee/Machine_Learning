class Perceptron:
    """ Class that defines a single Perceptron  with a list of the given weights an the input biases
    """
    def __init__(self, list_weights, biases):
        self.weights =list_weights
        self.biases = biases
        
    def activation(self, weighted_sum):
        """ Implements the step functon in order to define the the output of a percceptron

        Args:
            weighted_sum (Int): sum of the weights multiplied by the inputs.

        Returns:
            Bool: 0 or 1
        """
        #return 0 if weighted_sum < self.biases else 1 
        return 0 if (weighted_sum + self.biases)< 0 else 1
    # Bias = -threshold and the activation function wold return 0 if weighted_sum(weight *x + b)
    
    def calculate(self, inputs:[bool]):
        """Calculate the output of a perceptron given a list of inputs

        Args:
            inputs ([bool]): all possible combination of inputs of n x_inputs(x1, x2, etc)

        Returns:
            [type]: [description]
        """

        w_som = 0
        outputs = []
        for i in range(len(inputs)):# iterate through the index inputs e.g [0,0]
            weight = self.weights[i] # get weight 
            x = inputs[i] # get x

            w_som += (weight*x) # increment w_som with the multiplication of weighti and xi
        output = self.activation(w_som) # apply the step function to w_Som
            #print(outputs)
        return output
    
    def __str__(self):
        return ("Weights: {}" + "\n" + "Biase/Threshold {}").format(self.weights, self.biases)

class PerceptonLayer:
    
    def __init__(self):
        self.n_perceptrons = [] # list of Perceptrons objects
    
    #Setters and Getters
    def SetPerceptrons(self, p):
        self.n_perceptrons.append(p)
    
    def getPerceptrons(self):
        return self.n_perceptrons
    
    def get_output(self, input_arr):
        """Calculate the output for each perceptron in n_perceptron

        Returns:
            [int]: lsit of containing the outputs of each perceptron in 0 or 1
        """
        input_next_layer = []
        for p in self.n_perceptrons: # for each perceptron
            g = p.calculate(input_arr) # get the outpout 
            input_next_layer.append(g) # ad to list
        return input_next_layer


class PerceptonNetwork:
    
    def __init__(self):
        self.n_layers = []
        
    def setLayer(self, layer):
        self.n_layers.append(layer)
        
    def feed_forward(self, input_arr):
        for layer in self.n_layers: # for each layer
            input_next_layer = layer.get_output(input_arr) # calculate the outputs of each perceptron in layer
            input_arr = input_next_layer# set input_next_layer as input_arr to be used as input for the next layer
        return input_arr

