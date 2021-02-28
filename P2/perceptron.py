class Perceptron:
    """
    Class that defines a perceptron object
    """

    def __init__(self, list_weights, bias, constant_learning_rate):
        self.weights = list_weights
        self.bias = bias
        self.n = constant_learning_rate

        self.iter = 0
        self.sum_error = 0
        self.rmse = 0 

    def activation(self, weighted_sum):
        """ Implements the step functon in order to define the the output of a percceptron

        Args:
            weighted_sum (Int): sum of the weights multiplied by the inputs.

        Returns:
            Bool: 0 or 1
        """
        #return 0 if weighted_sum < self.bias else 1 
        return 0 if (weighted_sum + self.bias)< 0 else 1
    
    def get_output(self, input_arr):
        # calculate output  y = f(w * x)
        w_sum = 0
        for i in range(len(input_arr)): # for each input
            w_sum += input_arr[i] * self.weights[i]
        
        return self.activation(w_sum)

    def update(self, input_arr, output): 
        
        y = self.get_output(input_arr)
        # calculate error e = d - y
        e = y - output
        self.sum_error += e # Σ | d – y | voor error()

        # calculate new weights =>  w' = w + Δw
        for w in range(len(self.weights)):
            # calculate difference weight => Δw = η ∙ e ∙ x 
            diff_w = self.n * e * input_arr[w]
            self.weights[w] = self.weights[w] + diff_w

        # calculate new baise Δb = η ∙ e
        diff_b = self.n * e 
        # calculate new biase
        self.bias = self.bias + diff_b

        # TODO iter and error_sum documenteren 
        self.iter += 1


    def error(self):
        # MSE = Σ | d – y |^2 / n
        self.rmse(self.sum_error**2) /self.iter 
        
    def __str__(self):
        return ("Weights: {}" + "\n" + "Biase/Threshold {}").format(self.weights, self.bias)

