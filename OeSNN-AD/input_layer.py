from abstract_layer import Layer

class Input_Layer(Layer):
    
    def __init__(self, input_size, output_size) -> None:
        super().__init__(input_size, output_size)

    def _GRF_Initialization(window):
        ...