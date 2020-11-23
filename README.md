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


class RacingBranches(traits.NotValidated, Runnable):
    Runs (races) multiple workflows of type :class:`~hybrid.core.Runnable`
    in parallel, stopping all once the first finishes. Returns the results of
    all, in the specified order.

    Args:
        *branches ([:class:`~hybrid.core.Runnable`]):
            Comma-separated branches.

    Note:
        Each branch runnable is called with run option ``racing_context=True``,
        so it can adapt its behaviour to the context.

    Note:
        `RacingBranches` is also available as `Race`.

    Examples:
        This example runs two branches: a classical tabu search interrupted by
        samples of subproblems returned from a D-Wave system.

        ::

            RacingBranches(
                InterruptableTabuSampler(),
                EnergyImpactDecomposer(size=2)
                | QPUSubproblemAutoEmbeddingSampler()
                | SplatComposer()
            ) | ArgMin()


class InterruptableTabuSampler(Loop):
    An interruptable tabu sampler for a binary quadratic problem.

    Args:
        num_reads (int, optional, default=1):
            Number of states (output solutions) to read from the sampler.

        tenure (int, optional):
            Tabu tenure, which is the length of the tabu list, or number of
            recently explored solutions kept in memory. Default is a quarter of
            the number of problem variables up to a maximum value of 20.

        timeout (int, optional, default=20):
            Timeout for non-interruptable operation of tabu search. At the
            completion of each loop of tabu search through its problem
            variables, if this time interval has been exceeded, the search can
            be stopped by an interrupt signal or expiration of the `timeout`
            parameter.

        initial_states_generator (str, 'none'/'tile'/'random', optional, default='random'):
            Defines the expansion of input state samples into `initial_states`
            for the Tabu search, if fewer than `num_reads` samples are
            present. See :meth:`~tabu.TabuSampler.sample`.

        max_time (float, optional, default=None):
            Total running time in milliseconds.

class EnergyImpactDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    Selects a subproblem of variables maximally contributing to the problem
    energy.

    The selection currently implemented does not ensure that the variables are
    connected in the problem graph.

    Args:
        size (int):
            Nominal number of variables in the subproblem. Actual subproblem can
            be smaller, depending on other parameters (e.g. `min_gain`).

        min_gain (int, optional, default=-inf):
            Minimum reduction required to BQM energy, given the current sample.
            A variable is included in the subproblem only if inverting its
            sample value reduces energy by at least this amount.

        rolling (bool, optional, default=True):
            If True, successive calls for the same problem (with possibly
            different samples) produce subproblems on different variables,
            selected by rolling down the list of all variables sorted by
            decreasing impact.

        rolling_history (float, optional, default=1.0):
            Fraction of the problem size, as a float in range 0.0 to 1.0, that
            should participate in the rolling selection. Once reached,
            subproblem unrolling is reset.
            
        silent_rewind (bool, optional, default=True):
            If False, raises :exc:`EndOfStream` when resetting/rewinding the
            subproblem generator upon the reset condition for unrolling.

        traversal (str, optional, default='energy'):
            Traversal algorithm used to pick a subproblem of `size` variables.
            Options are:

            energy:
                Use the next `size` variables in the list of variables ordered
                by descending energy impact.

            bfs:
                Breadth-first traversal seeded by the next variable in the
                energy impact list.

            pfs:
                Priority-first traversal seeded by variables from the energy
                impact list, proceeding with the variable on the search boundary
                that has the highest energy impact.

class QPUSubproblemAutoEmbeddingSampler(traits.SubproblemSampler, traits.SISO, Runnable):
    A quantum sampler for a subproblem with automated heuristic
    minor-embedding.

    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.

        num_retries (int, optional, default=0):
            Number of times the sampler will retry to embed if a failure occurs.

        qpu_sampler (:class:`dimod.Sampler`, optional, default=\ :class:`~dwave.system.samplers.DWaveSampler`\ ``(client="qpu")``):
            Quantum sampler such as a D-Wave system. Subproblems that do not fit the
            sampler's structure are minor-embedded on the fly with
            :class:`~dwave.system.composites.AutoEmbeddingComposite`.

        sampling_params (dict):
            Dictionary of keyword arguments with values that will be used
            on every call of the (embedding-wrapped QPU) sampler.

        auto_embedding_params (dict, optional):
            If provided, parameters are passed to the
            :class:`~dwave.system.composites.AutoEmbeddingComposite` constructor
            as keyword arguments.

class SplatComposer(traits.SubsamplesComposer, traits.SISO, Runnable):

    A composer that overwrites current samples with subproblem samples.


class ArgMin(traits.NotValidated, Runnable):
    Selects the best state from a sequence of :class:`~hybrid.core.States`.

    Args:
        key (callable/str):
            Best state is judged according to a metric defined with a `key`.
            The `key` can be a `callable` with a signature::

                key :: (State s, Ord k) => s -> k

            or a string holding a key name/path to be extracted from the input
            state with `operator.attrgetter` method.

            By default, `key == operator.attrgetter('samples.first.energy')`,
            thus favoring states containing a sample with the minimal energy.

    Examples:
        This example runs two branches---a classical tabu search interrupted by
        samples of subproblems returned from a D-Wave system--- and selects the
        state with the minimum-energy sample::

            RacingBranches(
                InterruptableTabuSampler(),
                EnergyImpactDecomposer(size=2)
                | QPUSubproblemAutoEmbeddingSampler()
                | SplatComposer()
            ) | ArgMin()

@stoppable
class LoopUntilNoImprovement(traits.NotValidated, Runnable):
    Iterates :class:`~hybrid.core.Runnable` for up to `max_iter` times, or
    until a state quality metric, defined by the `key` function, shows no
    improvement for at least `convergence` number of iterations. Alternatively,
    maximum allowed runtime can be defined with `max_time`, or a custom
    termination Boolean function can be given with `terminate` (a predicate
    on `key`). Loop is always terminated on :exc:`EndOfStream` raised by body
    runnable.

    Args:
        runnable (:class:`~hybrid.core.Runnable`):
            A runnable that's looped over.

        max_iter (int/None, optional, default=None):
            Maximum number of times the `runnable` is run, regardless of other
            termination criteria. This is the upper bound. By default, an upper
            bound on the number of iterations is not set.

        convergence (int/None, optional, default=None):
            Terminates upon reaching this number of iterations with unchanged
            output. By default, convergence is not checked, so the only
            termination criteria is defined with `max_iter`. Setting neither
            creates an infinite loop.

        max_time (float/None, optional, default=None):
            Wall clock runtime termination criterion. Unlimited by default.

        key (callable/str):
            Best state is judged according to a metric defined with a `key`.
            `key` can be a `callable` with a signature::

                key :: (State s, Ord k) => s -> k

            or a string holding a key name/path to be extracted from the input
            state with `operator.attrgetter` method.

            By default, `key == operator.attrgetter('samples.first.energy')`,
            thus favoring states containing a sample with the minimal energy.

        terminate (callable, optional, default=None):
            Loop termination Boolean function (a predicate on `key` value)::

                terminate :: (Ord k) => k -> Bool