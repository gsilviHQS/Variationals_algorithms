from qiskit.opflow.gradients.circuit_gradients.lin_comb import *

class LinCombMod(LinComb):
    def __init__(self, img: bool = False):
        self._img = img
    def _gradient_states(
        self,
        state_op: StateFn,
        meas_op: Union[OperatorBase, bool] = True,
        target_params: Optional[Union[Parameter, List[Parameter]]] = None,
        open_ctrl: bool = False,
        trim_after_grad_gate: bool = False,
    ) -> ListOp:
        """Generate the gradient states.

        Args:
            state_op: The operator representing the quantum state for which we compute the gradient.
            meas_op: The operator representing the observable for which we compute the gradient.
            target_params: The parameters we are taking the gradient wrt: ω
            open_ctrl: If True use an open control for ``grad_gate`` instead of closed.
            trim_after_grad_gate: If True remove all gates after the ``grad_gate``. Can
                be used to reduce the circuit depth in e.g. computing an overlap of gradients.

        Returns:
            ListOp of StateFns as quantum circuits which are the states w.r.t. which we compute the
            gradient. If a parameter appears multiple times, one circuit is created per
            parameterized gates to compute the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
            TypeError: If the operators is of unsupported type.
        """
        qr_superpos = QuantumRegister(1)
        state_qc = QuantumCircuit(*state_op.primitive.qregs, qr_superpos)
        state_qc.h(qr_superpos)
        state_qc.compose(state_op.primitive, inplace=True)

        # Define the working qubit to realize the linear combination of unitaries
        if not isinstance(target_params, (list, np.ndarray)):
            target_params = [target_params]

        oplist = []
        for param in target_params:
            if param not in state_qc.parameters:
                oplist += [~Zero @ One]
            else:
                param_gates = state_qc._parameter_table[param]
                sub_oplist = []
                for gate, idx in param_gates:
                    grad_coeffs, grad_gates = self._gate_gradient_dict(gate)[idx]

                    # construct the states
                    for grad_coeff, grad_gate in zip(grad_coeffs, grad_gates):
                        grad_circuit = self.apply_grad_gate(
                            state_qc,
                            gate,
                            idx,
                            grad_gate,
                            grad_coeff,
                            qr_superpos,
                            open_ctrl,
                            trim_after_grad_gate,
                        )

                        # apply S if imaginary part needed
                        if self._img: grad_circuit.s(qr_superpos)

                        # apply final hadamard on superposition qubit
                        grad_circuit.h(qr_superpos)

                        # compute the correct coefficient and append to list of circuits
                        coeff = np.sqrt(np.abs(grad_coeff)) * state_op.coeff
                        state = CircuitStateFn(grad_circuit, coeff=coeff)

                        # apply the chain rule if the parameter expression if required
                        param_expression = gate.params[idx]

                        if isinstance(meas_op, OperatorBase):
                            state = meas_op @ state
                        elif meas_op is True:
                            state = ListOp(
                                [state], combo_fn=partial(self._grad_combo_fn, state_op=state_op)
                            )

                        if param_expression != param:  # parameter is not identity, apply chain rule
                            param_grad = param_expression.gradient(param)
                            state *= param_grad

                        sub_oplist += [state]

                oplist += [SummedOp(sub_oplist) if len(sub_oplist) > 1 else sub_oplist[0]]

        return ListOp(oplist) if len(oplist) > 1 else oplist[0]