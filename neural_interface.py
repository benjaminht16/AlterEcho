import alterecho

class AlterEchoInterface:
    """
    A class for controlling the AlterEcho neural interface for data input and output.
    """
    def __init__(self, connection_params):
        # Initialize connection to AlterEcho device
        self.connection_params = connection_params
        self.device = alterecho.connect(self.connection_params)
        self.is_connected = True
        
    def send_data(self, data):
        return alterecho.process(data, self.device)
        
    def receive_data(self):
        return alterecho.get_data(self.device)
        
    def disconnect(self):
        # Disconnect from AlterEcho device
        alterecho.disconnect(self.device)
        self.is_connected = False