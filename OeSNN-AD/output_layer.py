from abstract_layer import Layer

class Output_Layer(Layer):
    
    def __init__(self, input_size, output_size = 1) -> None:
        super().__init__(input_size, output_size)
