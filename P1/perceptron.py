class Perceptron:
    
    def __init__(self,inputsize, list_weights, biases):
        self.weights =list_weights
        self.biases = biases
        
    def activation(self, weighted_sum):
        #return 0 if weighted_sum < self.biases else 1 
        return 0 if (weighted_sum + self.biases)< 0 else 1
    # TODO implement biase and not threshold. 
    # Bias = -threshold and the activation function wold return 0 if weighted_sum(weight *x + b)
    
    def calculate(self, inputs:[bool]):
        w_som = 0
        outputs = []
        for row in range(len(inputs)):
            for i in range(len(inputs[row])):
                weight = self.weights[i] # get weight 
                x = inputs[row][i] # get x
                
                w_som += (weight*x)
            output = self.activation(w_som)
            outputs.append(output)
            w_som = 0
            #print(outputs)
        return outputs
    
    def __str__(self):
        return ("Weights: {}" + "\n" + "Biase/Threshold {}").format(self.weights, self.biases)

class PerceptonLayer:
    
    def __init__(self):
        self.n_perceptrons = []
        
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
                g = p.calculate(input_arr)
                input_next_layer.append(g)

            # TRANSPOSE A 2D ARRAY
            transposed_input_next_layer_arr = []
            for i in range(len(input_next_layer[0])):
                tmp = []
                for inputs in range(len(input_next_layer)):
                    tmp.append(input_next_layer[inputs][i])
                transposed_input_next_layer_arr.append(tmp)
            
            input_arr = transposed_input_next_layer_arr
        return input_arr


    # TODO 
    # finish test 
    # DOCUMENTATION 