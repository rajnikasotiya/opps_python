from abc import ABC, abstractmethod

# Abstract class
class Shape(ABC):
    
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass


# Concrete class inheriting from abstract class
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14 * self.radius


# Concrete class inheriting from abstract class
class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * (self.length + self.width)


# Creating Objects
circle = Circle(4)
rectangle = Rectangle(4, 7)

print(f"Circle Area: {circle.area()}")             # Output: Circle Area: 50.24
print(f"Circle Perimeter: {circle.perimeter()}")   # Output: Circle Perimeter: 25.12

print(f"Rectangle Area: {rectangle.area()}")       # Output: Rectangle Area: 28
print(f"Rectangle Perimeter: {rectangle.perimeter()}")  # Output: Rectangle Perimeter: 22
