class Complex:
    def __init__(self, re: float = 0, im: float = 0) -> None:
        self.re = re
        self.im = im

    def __add__(self, other):
        return Complex.add(self, other)

    def __mul__(self, other):
        return Complex.multiply(self, other)

    def __str__(self) -> str:
        if self.im < 0:
            return f"{self.re:.2f}{self.im:.2f}i"

        if self.im == 0:
            return str(self.re)

        return f"{self.re:.2f}+{self.im:.2f}i"

    @staticmethod
    def add(first, second):
        return Complex(first.re + second.re, first.im + second.im)

    @staticmethod
    def multiply(first, second):
        real_part = first.re * second.re - first.im * second.im
        imaginary_part = first.im * second.re + first.re * second.im

        return Complex(real_part, imaginary_part)
