from warnings import warn

class Mersenne:
    """Pseudorandom number generator"""

    def __init__(self, seed=1234):
        """
        Initialize pseudorandom number generator. Accepts an
        integer or floating-point seed, which is used in
        conjunction with an integer multiplier, k, and the
        Mersenne prime, j, to "twist" pseudorandom numbers out
        of the latter. This member also initializes the order
        of the generator's period, so that members floating and
        integer can emit a warning when generation is about to
        cycle and thus become not so pseudorandom.
        """
        self.seed = seed
        self.j = 2 ** 31 - 1
        self.k = 16807
        self.period = 2 ** 30

    def floating(self, interval=None, count=1):
        """
        Return a pseudorandom float. Default is one floating-
        point number between zero and one. Pass in a tuple or
        list, (a,b), to return a floating-point number on
        [a,b]. If count is 1, a single number is returned,
        otherwise a list of numbers is returned.
        """
        results = []
        for i in range(count):
            self.seed = (self.k * self.seed) % self.j
            if interval is not None:
                results.append(
                    (interval[1] - interval[0]) * (self.seed / self.j) + interval[0]
                )
            else:
                results.append(self.seed / self.j)
            self.period -= 1
            if self.period == 0:
                warn("Pseudorandom period nearing!!", category=ResourceWarning)
                self.period = 2 ** 30 # reset period
        if count == 1:
            return results.pop()
        else:
            return results

    def integer(self, interval=None, count=1):
        """
        Return a pseudorandom integer. Default is one integer
        number in {0,1}. Pass in a tuple or list, (a,b), to
        return an integer number on [a,b]. If count is 1, a
        single number is returned, otherwise a list of numbers
        is returned.
        """
        results = []
        for i in range(count):
            self.seed = (self.k * self.seed) % self.j
            if interval is not None:
                results.append(
                    int(
                        (interval[1] - interval[0] + 1) * (self.seed / self.j)
                        + interval[0]
                    )
                )
            else:
                result = self.seed / self.j
                if result < 0.50:
                    results.append(0)
                else:
                    results.append(1)
            self.period -= 1
            if self.period == 0:
                warn("Pseudorandom period nearing!!", category=ResourceWarning)
                self.period = 2 ** 30 # reset period
        if count == 1:
            return results.pop()
        else:
            return results