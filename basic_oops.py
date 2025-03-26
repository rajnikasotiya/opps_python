# Class Definition
class Shape:
    # Constructor to initialize the dimensions
    def __init__(self, color):
        self.color = color  # Encapsulating the color attribute

    # Method to display the color of the shape
    def display_color(self):
        print(f"The color of the shape is {self.color}")


# Creating Object (Instance of Shape)
circle = Shape("Red")
square = Shape("Blue")

# Accessing Methods
circle.display_color()    # Output: The color of the shape is Red
square.display_color()    # Output: The color of the shape is Blue
