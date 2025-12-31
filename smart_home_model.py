import copy

class SmartHomeState:
    def __init__(self):
        self.devices = {'Telewizor': False, 'Ciemniej': False, 'Jaśniej': False, 
                        'Światło': False, 'Muzyka': False, 'Rolety': False, 
                        'Wychodzę': False, 'Wróciłem': False}
        self.saved_state = None

    def toggle(self, name):
        if name in self.devices:
            self.devices[name] = not self.devices[name]
            return self.devices[name]
        return False
    
    def set_all(self, state):
        for s in self.devices:
            self.devices[s] = state
        return self.devices
    
    def save(self):
        self.saved_state = copy.deepcopy(self.devices)

    def restore(self):
        if self.saved_state:
            self.devices = copy.deepcopy(self.saved_state)
            return True
        return False