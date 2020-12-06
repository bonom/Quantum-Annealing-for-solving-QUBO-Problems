#!/usr/bin/env python3

class elements:
    i = 0
    f = 0
    integer_vector = list()
    float_vector = list()

    def __init__(self):
        try:
            with open("integer.txt", "r") as integers_to_read:
                for line in integers_to_read:
                    self.integer_vector.append(int(line))
        except:
            raise FileNotFoundError

        try:
            with open("float.txt", "r") as floats_to_read:
                for line in floats_to_read:
                    self.float_vector.append(float(line))
        except:
            raise FileNotFoundError

    def pop_int(self):
        integer_element = self.integer_vector[self.i]
        self.i = (self.i + 1) % 4000
        return integer_element

    def pop_float(self):
        float_element = self.float_vector[self.f]
        self.f = (self.f + 1) % 10000
        return float_element