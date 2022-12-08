class Derivative:
    @staticmethod
    def differentiate(func, a, h=1e-6):
        return (func(a + h) - func(a)) / (h)
