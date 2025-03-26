# Parent Class
class Shape:
    def __init__(self, color):
        self.color = color

    def display_color(self):
        print(f"The color of the shape is {self.color}")


# Child Class (Inheritance)
class Circle(Shape):
    def __init__(self, color, radius):
        super().__init__(color)   # Inheriting the color property
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, color, length, width):
        super().__init__(color)  # Inheriting the color property
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width


# Creating Objects of Child Classes
circle = Circle("Green", 5)
rectangle = Rectangle("Yellow", 4, 6)

# Accessing Inherited and Child Methods
circle.display_color()   # Output: The color of the shape is Green
print(f"Circle Area: {circle.area()}")     # Output: Circle Area: 78.5

rectangle.display_color()   # Output: The color of the shape is Yellow
print(f"Rectangle Area: {rectangle.area()}")  # Output: Rectangle Area: 24
