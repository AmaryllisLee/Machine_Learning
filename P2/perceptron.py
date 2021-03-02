class Perceptron:
    """
    Class that defines a perceptron object
    """

    def __init__(self, list_weights, bias, n_iters, constant_learning_rate = 0.1):
        self.weights = list_weights
        self.bias = bias
        self.n = constant_learning_rate
        self.n_iters = n_iters

    

    def activation(self, weighted_sum):
        """ Implements the step functon in order to define the the output of a percceptron

        Args:
            weighted_sum (Int): sum of the weights multiplied by the inputs.

        Returns:
            Bool: 0 or 1
        """
        #return 0 if weighted_sum < self.bias else 1 
        return 0 if (weighted_sum + self.bias)< 0 else 1

    
    def predict(self, input_arr):
        # calculate output  y = f(w * x)
        w_sum = 0
        for i in range(len(input_arr)): # for each input
            w_sum += input_arr[i] * self.weights[i]
        
        return self.activation(w_sum)

    def update(self, input_arr, output): 
        """ Implementation of a perceptron learning rul

        Args:
            input_arr: list of inputs in boolean for the perceptron 
            output : [description]
        """

        for _ in range(self.n_iters):
            y = self.predict(input_arr)
            # calculate error e = d - y
            e = y - output

            # calculate new weights =>  w' = w + Δw
            for w in range(len(self.weights)):
                # calculate difference weight => Δw = η ∙ e ∙ x 
                diff_w = self.n * e * input_arr[w]
                self.weights[w] = self.weights[w] + diff_w

            # calculate new baise Δb = η ∙ e
            diff_b = self.n * e 
            # calculate new biase
            self.bias = self.bias + diff_b
    
    # def error(self):
         # MSE = Σ | d – y |^2 / n

       
    def __str__(self):
        return ("Weights: {}" + "\n" + "Biase/Threshold {}").format(self.weights, self.bias)
    

# I would be using the Perceptorn layer from P1 becaus the class is not being modified
class PerceptronLayer:
    
    def __init__(self):
        self.n_perceptrons = [] # list of Perceptrons objects

    
    def activate(self, input_arr):
        """Calculate the output for each perceptron in n_perceptron

        Returns:
            [int]: lsit of containing the outputs of each perceptron in 0 or 1
        """
        input_next_layer = []
        for p in self.n_perceptrons: # for each perceptron
            g = p.predict(input_arr) # get the outpout 
            input_next_layer.append(g) # ad to list
        return input_next_layer

    def __str__(self):
        layer_size = len(self.n_perceptrons)
        perceptrons = [i.type_logic for i in self.n_perceptrons]
        return ( 'Size of layer is {} perceptrons' + '\n' + 'The perceptrons are {}').format(layer_size, perceptrons)


class PerceptonNetwork:
    
    def __init__(self):
        self.n_layers = []
    
        
    def feed_forward(self, input_arr):
        for layer in self.n_layers: # for each layer
            input_next_layer = layer.activate(input_arr) # calculate the outputs of each perceptron in layer
            input_arr = input_next_layer# set input_next_layer as input_arr to be used as input for the next layer
        return input_arr
    
   
    