# Quantum Annealing for solving QUBO Problems

class BinaryQuadraticModel(AdjDictBQM, Sized, Iterable, Container):
    Encodes a binary quadratic model.

    Binary quadratic model is the superclass that contains the `Ising model`_ and the QUBO_.

    .. _Ising model: https://en.wikipedia.org/wiki/Ising_model
    .. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    Args:
        linear (dict[variable, bias]):
            Linear biases as a dict, where keys are the variables of
            the binary quadratic model and values the linear biases associated
            with these variables.
            A variable can be any python object that is valid as a dictionary key.
            Biases are generally numbers but this is not explicitly checked.

        quadratic (dict[(variable, variable), bias]):
            Quadratic biases as a dict, where keys are
            2-tuples of variables and values the quadratic biases associated
            with the pair of variables (the interaction).
            A variable can be any python object that is valid as a dictionary key.
            Biases are generally numbers but this is not explicitly checked.
            Interactions that are not unique are added.

        offset (number):
            Constant energy offset associated with the binary quadratic model.
            Any input type is allowed, but many applications assume that offset is a number.
            See :meth:`.BinaryQuadraticModel.energy`.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the binary quadratic model. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        **kwargs:
            Any additional keyword parameters and their values are stored in
            :attr:`.BinaryQuadraticModel.info`.

    Notes:
        The :class:`.BinaryQuadraticModel` class does not enforce types on biases
        and offsets, but most applications that use this class assume that they are numeric.

    Examples:
        This example creates a binary quadratic model with three spin variables.

        >>> bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                  {(0, 1): .5, (1, 2): 1.5},
        ...                                  1.4,
        ...                                  dimod.Vartype.SPIN)

        This example creates a binary quadratic model with non-numeric variables
        (variables can be any hashable object).

        >>> bqm = dimod.BQM({'a': 0.0, 'b': -1.0, 'c': 0.5},
        ...                                  {('a', 'b'): -1.0, ('b', 'c'): 1.5},
        ...                                  1.4,
        ...                                  dimod.SPIN)
        >>> len(bqm)
        3
        >>> 'b' in bqm
        True

    Attributes:
        linear (dict[variable, bias]):
            Linear biases as a dict, where keys are the variables of
            the binary quadratic model and values the linear biases associated
            with these variables.

        quadratic (dict[(variable, variable), bias]):
            Quadratic biases as a dict, where keys are 2-tuples of variables, which
            represent an interaction between the two variables, and values
            are the quadratic biases associated with the interactions.

        offset (number):
            The energy offset associated with the model. Same type as given
            on instantiation.

        vartype (:class:`.Vartype`):
            The model's type. One of :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`.

        variables (keysview):
            The variables in the binary quadratic model as a dictionary keys
            view object.

        adj (dict):
            The model's interactions as nested dicts.
            In graphic representation, where variables are nodes and interactions
            are edges or adjacencies, keys of the outer dict (`adj`) are all
            the model's nodes (e.g. `v`) and values are the inner dicts. For the
            inner dict associated with outer-key/node 'v', keys are all the nodes
            adjacent to `v` (e.g. `u`) and values are quadratic biases associated
            with the pair of inner and outer keys (`u, v`).

        info (dict):
            A place to store miscellaneous data about the binary quadratic model
            as a whole.

        SPIN (:class:`.Vartype`): An alias of :class:`.Vartype.SPIN` for easier access.

        BINARY (:class:`.Vartype`): An alias of :class:`.Vartype.BINARY` for easier access.

    Examples:
       This example creates an instance of the :class:`.BinaryQuadraticModel`
       class for the K4 complete graph, where the nodes have biases
       set equal to their sequential labels and interactions are the
       concatenations of the node pairs (e.g., 23 for u,v = 2,3).

       >>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
       >>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
       ...              (2, 3): 23, (2, 4): 24,
       ...              (3, 4): 34}
       >>> offset = 0.0
       >>> vartype = dimod.BINARY
       >>> bqm_k4 = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
       >>> len(bqm_k4.adj[2])            # Adjacencies for node 2
       3
       >>> bqm_k4.adj[2][3]         # Show the quadratic bias for nodes 2,3
       23