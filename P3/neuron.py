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
       

    def gradient(self):
        "Bereken de gradient van een gewicht tussen neuron i en output neuron j."
        return self.output * self.error

    def update(self, input_arr):
        "Update the weights and bias for 1 iteration "
        for i in range(len(self.weights)):
            # print(round(self.weights[i] - (self.n * input_arr[i] * self.gradient()), 3)) #TODO The results in the notebook is exact to the one in the werkboek. Is because the werkboek is rounding up or it is an error in the code? - CHECK LATER
            self.weights[i] = self.weights[i] - (self.n * input_arr[i]* self.gradient()) # w'i,j = wi,j – Δwi,j, Δwi,j = η ∙ ∂C/∂wi,j = η ∙ outputi ∙ Δj
        
        self.bias = self.bias - self.n * self.error # b'j = bj – Δbj, Δbj = η ∙ Δj
    
    def cost(self, ):
        "Calculating the total loss by using the cost function Mean Squared error"
        # Formula of the MSE =  Σ |d - y|/n # TODO example of cost

    

    def __str__(self):
        return ("Weights: {}" + "\n" + "Biase/Threshold {}").format(self.weights, self.bias)


class HiddenNeuron(Neuron):
    
    def __init__(self, list_weights, bias, constant_learning_rate=0.1):
        super().__init__(list_weights, bias, constant_learning_rate=constant_learning_rate)

    def set_error(self, weights_next_layer, error_next_layer):
        "Bereken de error van een hidden layer"
        errors =  [(weights_next_layer[i] * error_next_layer[i]) for i in range(weights_next_layer)]
        self.error = self.output * ( 1- self.output) * errors
       

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
        for p in self.n_neurons: # for each perceptron
            input_next_layer.append(p.calculate_output(input_arr)) # get the outpout
        return input_next_layer 
    
    def set_error_output(self, d):
        for n in self.n_neurons:
            n.set_error(d)
            self.errors.append(n.error)
    
    def set_error_hidden(self, next_layer):
        for neuron in next_layer.neurons:
            neuron.set_error()
            self.errors.append(neuron.error)

        
    
    def update_layer(self, input_arr):
        for n in self.n_neurons:
            n.update(input_arr)

    def __str__(self):
        layer_size = len(self.n_neurons)
        return ( 'Size of layer is {} perceptrons').format(layer_size)


class NeuronNetwork:

    def __init__(self):
        self.n_layers = []
        self.a = [] # list containing the output of the neurons: e.g ao, af,ag,am, an

    def feed_forward(self, input_arr):
        for layer in self.n_layers: # for each layer
            input_next_layer = layer.get_output(input_arr) # calculate the outputs of each perceptron in layer
            self.a.append(input_next_layer)
            input_arr = input_next_layer# set input_next_layer as input_arr to be used as input for the next layer
        # return input_arr
    
    def set_error(self):
        for layer in self.n_layers:
            for neuron in layer.n_neurons:
                neuron.error = 0
    
    def MSE(self, training_examples,  targets):
        loss=0
        loss_lst = []
        for example   in training_examples:
            self.feed_forward(example)
            for i in range(len(targets)):
                loss  +=  (targets[i] - self.a[i])**2
            loss_lst.append(loss/len(targets))
        return sum(loss_lst)/len(loss_lst)
            


    def train(self, training_examples, targets, epochs= 1000):
        "Implementin backpropagation"
        epoch = 0
        while  epoch <   epochs: #iterating through 100 epoch  TODO change for stopcoondition
            for example in training_examples:
                #feedforward
                self.feed_forward(example)

                # set errors
                errors  = []
                for i in range(len(reversed(self.layers))):
                    if i == 0: # check if layer is outputlayer
                        self.n_layers[-1].set_error_output(targets[i])
                    self.n_layers[i-1].set_error_hidden(self.n_layers[i])

                # Update
                for layer in reversed(self.layers):
                    layer.update(example)
            epoch +=1
            #cost
            print(f"C(w) =  {self.MSE(training_examples, targets)}")

#
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
        

        

    
    

