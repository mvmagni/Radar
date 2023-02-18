
class BoundingBox:
   
    def __init__(self, 
                top_left_x, 
                top_left_y,
                width,
                height):
        
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.width = width
        self.height = height

    def get_top_left(self):
        return self.top_left_x, self.top_left_y
    
    def get_top_right(self):
        return self.top_left_x + self.width, self.top_left_y
    
    def get_bottom_right(self):
        return self.top_left_x + self.top_left_y + self.height
    
    def get_bottom_right(self):
        return self.top_left_x + self.width, self.top_left_y + self.height
    
    def get_centre(self):
        return int(round(self.top_left_x + self.width / 2, 0)), int(round(self.top_left_y+self.height/2, 0))
    
    