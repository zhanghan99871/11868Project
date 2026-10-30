from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import deque 


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    
    Hints:
        1. Ensure that you visit the computation graph in a post-order depth-first search.
        2. When the children nodes of the current node are visited, add the current node 
            at the front of the result order list.
    """
    # BEGIN HW2_1
    # TODO
    result = deque() 
    visit = set() 
    def dfs(variable: Variable):
        if variable.unique_id in visit:
            return 
        if variable.is_constant():
            return 
        visit.add(variable.unique_id) 
        if not variable.is_leaf():
            for each in variable.parents:
                dfs(each) 
        result.appendleft(variable)
    
    dfs(variable) 
    return result 
            
    # raise NotImplementedError("Task Autodiff Not Implemented Yet")
    # END HW2_1


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write its results to the derivative values of each leaf through `accumulate_derivative`.

    Hints:
        1. Traverse nodes in topological order
        2. If the node is a leaf, the derivative should be accumulated
        3. Otherwise, the derivative should be propagated via chain rule
    """
    # BEGIN HW2_1
    # TODO
    record = {variable.unique_id:deriv} 
    topo_list = topological_sort(variable) 
    for v in topo_list:
        if v.is_leaf():
            v.accumulate_derivative(record[v.unique_id]) 
        else:
            for parent, p_grad in v.chain_rule(record[v.unique_id]):
                if parent.is_constant():
                    continue
                record[parent.unique_id] = record.get(parent.unique_id, 0) + p_grad 
    
    # raise NotImplementedError("Task Autodiff Not Implemented Yet")
    # END HW2_1


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
