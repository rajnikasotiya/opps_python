class Shape:
    def area(self, length=0, width=0):
        if length and width:
            return length * width  # Rectangle
        elif length:
            return length * length  # Square
        else:
            return 0


s = Shape()
print(s.area(5))            # Output: 25 (Square)
print(s.area(5, 10))        # Output: 50 (Rectangle)
print(s.area())             # Output: 0
