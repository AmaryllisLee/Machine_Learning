class Perceptron:
    """ Class that defines a single Perceptron  with a list of the given weights an the input biases
    """
    def __init__(self,inputsize, list_weights, biases):
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
        for row in range(len(inputs)):
            for i in range(len(inputs[row])):
                weight = self.weights[i] # get weight 
                x = inputs[row][i] # get x
                
                w_som += (weight*x) # increment w_som with the multiplication of weighti and xi
            output = self.activation(w_som) # apply the step function to w_Som
            outputs.append(output) # ad output ot list outputs
            w_som = 0 # reset w_Som to 0
            #print(outputs)
        return outputs
    
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

class PerceptonNetwork:
    
    def __init__(self):
        self.n_layers = []
        
    def setLayer(self, layer):
        self.n_layers.append(layer)
        
    def feed_forward(self, input_arr):
        for layer in self.n_layers: # for each layer
            input_next_layer = []
            for p in layer.getPerceptrons(): # for each perceptron
                g = p.calculate(input_arr) # get the outpout 
                input_next_layer.append(g) # ad to list

            # TRANSPOSE input_next_layer
            transposed_input_next_layer_arr = []
            for i in range(len(input_next_layer[0])):
                tmp = []
                for inputs in range(len(input_next_layer)):
                    tmp.append(input_next_layer[inputs][i])
                transposed_input_next_layer_arr.append(tmp)
            
            input_arr = transposed_input_next_layer_arr # add as input_arr to be used as input for the next layer
        return input_arr

