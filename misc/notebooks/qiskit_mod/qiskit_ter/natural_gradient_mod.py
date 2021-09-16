from qiskit.opflow.gradients.natural_gradient import *

class NewNaturalGradient(NaturalGradient):
    def convert(
            self,
            operator: OperatorBase,
            params: Optional[
                Union[ParameterVector, ParameterExpression, List[ParameterExpression]]
            ] = None,
        ) -> OperatorBase:
            r"""
            Args:
                operator: The operator we are taking the gradient of.
                params: The parameters we are taking the gradient with respect to. If not explicitly
                    passed, they are inferred from the operator and sorted by name.

            Returns:
                An operator whose evaluation yields the NaturalGradient.

            Raises:
                TypeError: If ``operator`` does not represent an expectation value or the quantum
                    state is not ``CircuitStateFn``.
                ValueError: If ``params`` contains a parameter not present in ``operator``.
                ValueError: If ``operator`` is not parameterized.
            """
            if not isinstance(operator, ComposedOp):
                if not (isinstance(operator, ListOp) and len(operator.oplist) == 1):
                    raise TypeError(
                        "Please provide the operator either as ComposedOp or as ListOp of "
                        "a CircuitStateFn potentially with a combo function."
                    )

            if not isinstance(operator[-1], CircuitStateFn):
                raise TypeError(
                    "Please make sure that the operator for which you want to compute "
                    "Quantum Fisher Information represents an expectation value or a "
                    "loss function and that the quantum state is given as "
                    "CircuitStateFn."
                )
            if len(operator.parameters) == 0:
                raise ValueError("The operator we are taking the gradient of is not parameterized!")
            if params is None:
                params = sorted(operator.parameters, key=functools.cmp_to_key(_compare_parameters))
            if not isinstance(params, Iterable):
                params = [params]
            # Instantiate the gradient
            grad = Gradient(self._grad_method, epsilon=self._epsilon).convert(operator, params)
            # Instantiate the QFI metric which is used to re-scale the gradient
            metric = self._qfi_method.convert(operator[-1], params) * 0.25

            # Define the function which compute the natural gradient from the gradient and the QFI.
            def combo_fn(x):
                c = -np.real(x[0]) #the modifcation is here, just a minus sign!
                a = np.real(x[1])
                #print(a,'\n',c)
                if self.regularization:
                    # If a regularization method is chosen then use a regularized solver to
                    # construct the natural gradient.
                    nat_grad = NaturalGradient._regularized_sle_solver(
                        a, c, regularization=self.regularization
                    )
                else:
                    try:
                        # Try to solve the system of linear equations Ax = C.
                        nat_grad = np.linalg.solve(a, c)
                    except np.linalg.LinAlgError:  # singular matrix
                        nat_grad = np.linalg.lstsq(a, c)[0]
                return np.real(nat_grad)

            # Define the ListOp which combines the gradient and the QFI according to the combination
            # function defined above.
            return ListOp([grad, metric], combo_fn=combo_fn)
    pass