# Parent Class
class Shape:
    def area(self):
        print("Area calculation is not defined for generic shape")


# Child Classes
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    # Overriding area method
    def area(self):
        return 3.14 * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    # Overriding area method
    def area(self):
        return self.length * self.width


# Function to demonstrate polymorphism
def print_area(shape):
    print(f"Area: {shape.area()}")


# Creating Objects
circle = Circle(7)
rectangle = Rectangle(5, 8)
shape = Shape()

# Polymorphism in action
print_area(circle)       # Output: Area: 153.86
print_area(rectangle)    # Output: Area: 40
print_area(shape)        # Output: Area calculation is not defined for generic shape
