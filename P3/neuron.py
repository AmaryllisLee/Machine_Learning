class Neuron:

    def __init__(self, list_weights, bias, constant_learning_rate = 0.1):
        self.weights = list_weights
        self.bias    = bias
        self.n = constant_learning_rate
      


        self.output = 0
        self.error = 0
        self.total_error= 0
    



    def activation(self,  product_weights):
        return 1/(1 + 2.71828**(-product_weights - self.bias)) # sigmoid formule , e = 2.71828


    def calculate_output(self, input_arr):
        w_som = 0
        for i in range(len(input_arr)):# iterate through the index inputs e.g [0,0]
            weight = self.weights[i] # get weight 
            x = input_arr[i] # get x

            w_som += (weight*x) # increment w_som with the multiplication of weighti and xi
        self.output = self.activation(w_som) # apply the step function to w_Som
        return self.output
       

    def gradient(self):
        "Bereken de gradient van een gewicht tussen neuron i en output neuron j."
        return self.output * self.error

    def update(self, input_arr):
        "Update the weights and bias for 1 iteration "
        for i in range(len(self.weights)):
            # print(round(self.weights[i] - (self.n * input_arr[i] * self.gradient()), 3)) #TODO The results in the notebook is exact to the one in the werkboek. Is because the werkboek is rounding up or it is an error in the code? - CHECK LATER
            self.weights[i] = self.weights[i] - (self.n * input_arr[i]* self.gradient()) # w'i,j = wi,j – Δwi,j, Δwi,j = η ∙ ∂C/∂wi,j = η ∙ outputi ∙ Δj
        
        self.bias = self.bias - self.n * self.error # b'j = bj – Δbj, Δbj = η ∙ Δj
    
    def cost(self, input_arr, targets):
        "Calculating the total loss of a neuron by using the cost function Mean Squared error"
        # Formula of the MSE =  Σ |d - y|/n # TODO example of cost
        loss = 0
        for i  in input_arr:
            output = self.calculate_output(i)
            for target in targets:
                loss += (target - output)**2
        return loss/ len(input_arr) 
            

    

    def __str__(self):
        return ("Weights: {}" + "\n" + "Biase/Threshold {}").format(self.weights, self.bias)


class HiddenNeuron(Neuron):
    
    def __init__(self, list_weights, bias, constant_learning_rate=0.1):
        super().__init__(list_weights, bias, constant_learning_rate=constant_learning_rate)

    def set_error(self, weights_next_layer, error_next_layer):
        "Bereken de error van een hidden layer"
        errors_sum =  sum([(weights_next_layer[i] * error_next_layer[i]) for i in range(len(weights_next_layer))])
        self.error = self.output * ( 1- self.output) * errors_sum
       

class OutputNeuron(Neuron):
    def __init__(self, list_weights, bias, constant_learning_rate=0.1):
        super().__init__(list_weights, bias, constant_learning_rate=constant_learning_rate)
    
    def set_error(self, d):
        """Bereken de error van de output neuron
        De formule voor de error luidt als volgt:
        o' = a0 * (1-a0) * - (d -  a0)
        a0 : output van de neuron met de feedforward/calculate_output
        d : verwachte output
        """
        self.error = self.output * (1 - self.output) * - (d - self.output)


class NeuronLayer:

    def __init__(self):
        self.n_neurons = [] # list of Neuron objects
        self.errors = [] # list that contains the erros of the neunronns

    def get_output(self, input_arr):
        """Calculate the output for each perceptron in n_perceptron

        Returns:
            [int]: lsit of containing the outputs of each perceptron in 0 or 1
        """
        input_next_layer = []
        for n in self.n_neurons: # for each neuron
            input_next_layer.append(n.calculate_output(input_arr)) # get the outpout
        return input_next_layer 
    

    def __str__(self):
        layer_size = len(self.n_neurons)
        return ( 'Size of layer is {} perceptrons').format(layer_size)

class HiddenLayer(NeuronLayer):
    
    def __init__(self):
        super().__init__()

    def set_errors(self, next_layer):
        "Bereken de error van een hidden layer"
        #set neurons
        for index in range(len(self.n_neurons)):
            layer_weights_i = [n.weights[index] for n in next_layer.n_neurons] # get the weights of the following layer
            # Set error for neuron "index"
            self.n_neurons[index].set_error(layer_weights_i, next_layer.errors)

        self.errors = [neuron.error for neuron in self.n_neurons]

class OutputLayer(NeuronLayer):
    def __init__(self):
        super().__init__()
    
    def set_errors(self, d):
        """Bereken de error van de outputlayer
        """
        #Set neurons 
        for i, neuron in enumerate(self.n_neurons):
            neuron.set_error(d[i])

        self.errors = [neuron.error for neuron in self.n_neurons]


class NeuronNetwork:

    def __init__(self):
        self.n_layers = []

    def feed_forward(self, input_arr):
        for layer in self.n_layers: # for each layer
            input_next_layer = layer.get_output(input_arr) # calculate the outputs of each perceptron in layer
            input_arr = input_next_layer# set input_next_layer as input_arr to be used as input for the next layer
        return input_arr
    
    def set_errors(self, d):
        for index, layer in enumerate(self.n_layers[::-1]): # iterating through layer from a descending order ( reverse n_layers )
            if index == 0: # check if layer is outputLayer
                layer.set_errors(d)
            else:
                layer.set_errors(self.n_layers[::-1][index-1])

    def update(self, training_example):
        inputs = training_example
        for layer in self.n_layers:
            for neuron_layer in layer.n_neurons:
                neuron_layer.update(inputs)
            inputs = [n.output for n in layer.n_neurons]


    def cost(self, training_examples,  targets):
        loss=0
        for example   in training_examples:
            output = self.feed_forward(example)
            for target in targets:
                for i in range(len(target)): 
                    loss  +=  (target[i] - output[i])**2
        return loss/len(training_examples)
            


    def train(self, training_examples, targets, epochs= 1000):
        "Implementin backpropagation"
        epoch = 0
        while  epoch <  epochs: #iterating through 1000 epoch  TODO change for stopcoondition
            for index, example in enumerate(training_examples): # iterating through each training example
                self.feed_forward(example) # predict the outputs of each layer with feedforward
                self.set_errors(targets[index])
                self.update(example) 
            epoch +=1
        print(f"C(w) =  {self.cost(training_examples, targets)}")


        
#  TODO : make adjustments to MSE, train ( specifically the set errors  part)
#  TODO (goal):  Mulitilayer test run and i correct 
# TODO :   implement part 18 and 19 of assignment: classification of the iris and the Digit dataset


        




        
    # def train(self,training_examples):
    #     cost = None
    #     while cost < 0.1:
    #         for example in training_examples:
    #             self.feed_forward(training_examples)
    #             self.set_error()
    #             self.update()
            
    #         cost = self.MSE(training_examples) 
        

        

    
    

