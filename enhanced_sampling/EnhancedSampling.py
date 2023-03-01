"""
Enhanced sampling module.
"""
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli

from qiskit.opflow import MatrixOp, StateFn, PauliExpectation, PrimitiveOp, PauliOp
from qiskit.opflow.state_fns import CircuitStateFn
from qiskit.opflow.converters import CircuitSampler

import numpy as np
import scipy
import matplotlib.pyplot as plt
import collections
import functools
from scipy.optimize import curve_fit

from typing import Optional, Tuple, Dict
import time
from random import random, seed
from math import pi

class EnhancedSampler():
    """
    The class used for enhanced sampling.
    """

    def __init__(
        self,
        Hamiltonian: PauliSumOp,
        Layers: int,
        ansatz,
        x_angles: Optional[np.ndarray] = None,
        binning_range: Tuple[float, float] = (-1, 1),
        binning_points: int = 10000,
    ) -> None:
        """
        Initialize the Enhanced sampling module.

        Args:
            Hamiltonian: the Hamiltonian to sample.
            Layers: the layers of the circuit to use to sample.
            ansatz: the ansatz to use.
            x_angles: the initial angles for the ansatz.
            binning_range: the range for the binning arrays.
            binning_points: the number of points in the binning arrays.
        """
        # Make circuits
        self._hamiltonian = Hamiltonian
        self._num_qubits = Hamiltonian.num_qubits
        print('num qubits',self._num_qubits)
        self._layers = Layers
        if ansatz.num_parameters == 0:
            self._ansatz = ansatz
        else:
            print("Ansatz with parameters not set, setting it random")
            seed(44)
            angles_set = {par: random()*2*pi for par in ansatz.ordered_parameters}
            print(angles_set)
            self._ansatz = ansatz.bind_parameters(angles_set)

        self._binning_points = binning_points
        self._x_angles = x_angles or np.ones(2 * Layers) * np.pi / 2

        binning_start, binning_end = binning_range
        self._binning = np.linspace(binning_start, binning_end, self._binning_points)
        self._binning_theta = np.linspace(0, np.pi, self._binning_points)

    def eval(self,
             pre_energy_freq: Dict[str, Dict[int, Dict[int, float]]],
             std_dev_freq: Dict[str, Dict[int, Dict[int, float]]],
             q_instance_post,
             repetitions: int = 1,
             steps: int = 1,
             steps_pre: int = 1,
             ) -> Tuple[Dict[str, Dict[int, Dict[int, float]]], Dict[str, Dict[int, Dict[int, float]]]]:
        # Save the standard x angles and layers
        standard_x_angles = self._x_angles
        standard_layers = self._layers
        # Create empty dictionaries for the likelihood, fitted energy, and fitted variance
        likelihood = {}
        fit_energy = {}
        fit_variance = {}

        # Loop through each part of the Hamiltonian
        for H_part in self._hamiltonian.to_pauli_op():
            # Convert the Hamiltonian part to a string
            primitive = str(H_part)
            print("\n\nSampling for", primitive, "...")
            # Create empty dictionaries for the fitted energy and variance for this Hamiltonian part
            fit_energy[primitive] = {}
            fit_variance[primitive] = {}
            for step in range(steps):
                fit_energy[primitive][step] = {}
                fit_variance[primitive][step] = {}

            # Reset the x angles and layers to the standard values
            self._x_angles = standard_x_angles
            self._layers = standard_layers

            # Loop through each repetition, useful for averaging plots, but not really necessary in practice
            
            for rep in range(repetitions):
                # Convert the pre-sampled energy frequency and standard deviation frequency to Theta
                start_time = time.time()
                self.convert_to_Theta(pre_energy_freq[primitive][steps_pre - 1][rep],
                                      std_dev_freq[primitive][steps_pre - 1][rep])

                # Compute the likelihood and Fisher information for the initial theta values
                if rep == 0:
                    likelihood_0, likelihood_1, ket_A = self._compute_likelihood(H_part)

                f_info_A = self.FischerInfo(self._initial_theta, ket_A)

                

                # Check if using 1 layer less results in a higher Fisher information
                if self._layers > 1:
                    self._layers -= 1
                    self._x_angles = self._x_angles[:-2]
                    if rep == 0:  # only compute likelihood if it hasn't been computed yet
                        likelihood_0_B, likelihood_1_B, ket_B = self._compute_likelihood(H_part)
                    f_info_B = self.FischerInfo(self._initial_theta, ket_B)

                    # Use the alternative circuit if it results in a higher Fisher information
                    if f_info_A >= f_info_B:
                        self._layers = standard_layers
                        self._x_angles = standard_x_angles
                        likelihood[0] = likelihood_0
                        likelihood[1] = likelihood_1
                    else:
                        print(' use alternative circuit')
                        likelihood[0] = likelihood_0_B
                        likelihood[1] = likelihood_1_B
                else:
                    likelihood[0] = likelihood_0
                    likelihood[1] = likelihood_1

                # Create the enhanced sampling circuit
                circuit = self.make_enhanced_circuit(H_part)  # TODO: check if it can be moved outside the loop
                end_time = time.time()
                print('Time pre-sampling:', end_time - start_time)
                # Initialize the outcomes dictionary
                outcomes = {0: 0, 1: 0}
                # Loop through each step
                start_time = time.time()
                for step in range(steps):
                    # Sample from the enhanced sampling circuit and update the outcomes
                    # at each step new outcomes are added to the dictionary
                    sampler_enhanced = CircuitSampler(backend=q_instance_post, attach_results=True).convert(circuit)
                    outcomes = self.collect_events(sampler_enhanced, outcomes)
                    # Compute the fitted energy and variance for this step
                    energy, variance = self.compute_posterior(outcomes, likelihood)
                    fit_energy[primitive][step][rep] = energy
                    fit_variance[primitive][step][rep] = variance
                    # Divide the fitted energy and variance by the number of repetitions
                    # for step in range(steps):
                    #     fit_energy[primitive][step] /= repetitions
                    #     fit_variance[primitive][step] /= repetitions
                end_time = time.time()
                print('Time sampling:', end_time - start_time)

        # Return the fitted energy and variance dictionaries
        return fit_energy, fit_variance

    def make_enhanced_circuit(self, Pauli_H):
        # Get the inverse of the ansatz
        ansz_inverse = self._ansatz.inverse()
        # Set the phase flip operator
        flip_op = -np.identity(2 ** self._num_qubits)
        flip_op[0, 0] = 1
        phase_flip_op = MatrixOp(flip_op)

        # Initialize quantum circuit
        list_of_qubits = list(range(self._num_qubits))
        circuit = QuantumCircuit(self._num_qubits)
        circuit.append(self._ansatz, list_of_qubits)

        # Add U and V gates to the circuit with angles bound
        for i, x in enumerate(self._x_angles):
            if i % 2 == 0:  # add U gate
                U_gate = (x * Pauli_H).exp_i().to_matrix()
                circuit.append(MatrixOp(U_gate), list_of_qubits)
            elif i % 2 == 1:  # add V gate
                R0_gate = (x * phase_flip_op).exp_i().to_matrix()
                circuit.append(ansz_inverse, list_of_qubits)
                circuit.append(MatrixOp(R0_gate), list_of_qubits)
                circuit.append(self._ansatz, list_of_qubits)

        # Compute the projection operator for the Hamiltonian
        proj_H_m = (PauliOp(Pauli('I' * self._num_qubits), coeff=1.0) - Pauli_H) / 2

        # Compose the circuit with the projection operator
        complete_circuit_m = StateFn(proj_H_m, is_measurement=True).compose(CircuitStateFn(primitive=circuit))

        # Compute the expectation value using the Pauli expectation
        expectation_m = PauliExpectation().convert(complete_circuit_m)

        # Return the expectation value
        return expectation_m

    def compute_posterior(self, outcomes, likelihood):
        # Compute the posterior distribution for theta
        print("\n\n NEW FIT")
        def get_posterior(likelihood, f_prior):
            estimate = sum(likelihood * f_prior)
            return (likelihood * f_prior) / estimate

        prior_theta = self._initial_prior_theta
        outcomes0 = outcomes[0]
        outcomes1 = outcomes[1]
        for _ in range(outcomes[0]+outcomes[1]):
            for outcome in [0,1]:
                if outcome == 0 and outcomes0 > 0:
                    outcomes0-=1
                    prior_theta = get_posterior(likelihood[outcome], prior_theta)
                elif outcome == 1 and outcomes1 > 0:
                    prior_theta = get_posterior(likelihood[outcome], prior_theta)
                    outcomes1-=1


        # for state, samples in outcomes.items(): #state can be 0 or 1, samples is the number of times 0 or 1 was measured
        #     for _ in range(samples):  # update the prior_theta for each sample, 
        #         prior_theta = get_posterior(likelihood[state], prior_theta)

        self._final_prior_theta = prior_theta

        # Guesses for the fit on the theta distribution
        total_counts = sum(prior_theta)
        mean_guess = sum(self._restricted_binning_theta * prior_theta) / total_counts
        std_dev_guess = np.sqrt(abs(sum(prior_theta * (self._restricted_binning_theta - mean_guess) ** 2) / total_counts))

        # Define the Gaussian function
        def gaussian_function(x, amplitude, mean, std_dev):
            return amplitude * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))
        
        # Set dynamic initial guess for amplitude
        amplitude_guess = np.max(prior_theta)
        

 
        print('Outcomes:', outcomes)
        if False:
            plt.plot(self._restricted_binning_theta, self._initial_prior_theta, label='initial prior')
            plt.plot(self._restricted_binning_theta, self._final_prior_theta, label='final prior')
            # plt.plot(self._restricted_binning_theta, likelihood[0], label='likelihood 0 with n outcomes'+str(outcomes[0]))
            # plt.plot(self._restricted_binning_theta, likelihood[1], label='likelihood 1 with n outcomes'+str(outcomes[1]))
            plt.legend()
            plt.show()
        # which is the same as
        
        
        
        print("Guesses: amp, mean and std_dev:",amplitude_guess, mean_guess, std_dev_guess)
        

        # Fit the theta distribution to a Gaussian
        try:
            popt, _ = curve_fit(gaussian_function, 
                                self._restricted_binning_theta, 
                                prior_theta, 
                                p0=[amplitude_guess, mean_guess, std_dev_guess])
            _, theta_fit, sigma_fit = popt
            print('FIT SUCCESS: Converged with popt:', popt)
        except RuntimeError:
            # plot the prior_theta
            print('\033[91m' + 'FIT ERROR: Failde to converge' + '\033[0m') # to make it red, use 
            plt.plot(self._restricted_binning_theta, self._initial_prior_theta, label='initial prior')
            plt.plot(self._restricted_binning_theta, self._final_prior_theta, label='final prior')
            plt.legend()
            plt.show()
            
            theta_fit, sigma_fit = mean_guess, std_dev_guess


        # self._theta_fit = theta_fit
        # self._sigma_fit = sigma_fit

        # Convert the fitted theta values to energy
        fit_energy, fit_variance = self._convert_to_energy(theta_fit, sigma_fit)

        return fit_energy, fit_variance

    def _convert_to_energy(self, mu, sigma):
        # Convert the mean and standard deviation of the theta distribution to energy
        energy = np.exp((-sigma ** 2) / 2) * np.cos(mu)
        variance = (1 - np.exp(-sigma ** 2)) * (1 - np.exp(-sigma ** 2) * np.cos(2 * mu)) / 2
        return energy, variance

    def convert_to_Theta(self, initial_mean, initial_std_dev) -> None:
        # Compute initial distribution and sample from it
        initial_distribution = scipy.stats.norm.pdf(self._binning, initial_mean, initial_std_dev)
        s = np.arccos(np.random.normal(initial_mean, initial_std_dev, self._binning_points * 100))

        # Compute initial prior for theta
        (initial_theta, initial_sigma) = scipy.stats.norm.fit([x for x in s if str(x) != 'nan'])
        if initial_sigma ** 2 > 0.01:
            print("WARNING: Large variance")
        initial_prior = scipy.stats.norm.pdf(self._binning_theta, initial_theta, initial_sigma)
        # limit the initial prior to the region of interest, i.e. from - 4 sigma to + 4 sigma
        self._binning_bounds = [np.argmax(initial_prior) - 4 * int(initial_sigma * self._binning_points), np.argmax(initial_prior) + 4 * int(initial_sigma * self._binning_points)]
        print("Initial prior bounds:", self._binning_bounds)
        initial_prior = initial_prior[self._binning_bounds[0]:self._binning_bounds[1]]
        initial_prior /= sum(initial_prior)
        self._restricted_binning_theta = self._binning_theta[self._binning_bounds[0]:self._binning_bounds[1]]
        # Save results to class variables
        self._sampling = s
        self._initial_distribution = initial_distribution
        self._initial_prior_theta = initial_prior
        self._initial_theta = initial_theta
        self._initial_sigma = initial_sigma

        return

    def _plot_initial_distribution(self) -> None:
        # Plot the initial distribution
        plt.title('Distribution on -1,1 values')
        plt.plot(self._binning, self._initial_distribution)

        # Plot the sampling from the distribution
        plt.figure()
        count, bins, ignored = plt.hist(self._sampling, 1000, density=True)
        plt.title('Conversion to Theta, sampling from distribution')
        plt.plot(bins, 1 / (self._initial_sigma * np.sqrt(2 * np.pi))
                 * np.exp(- (bins - self._initial_theta)**2 / (2 * self._initial_sigma**2)),
                 linewidth=2, color='r')
        plt.show()
        return

    def collect_events(self, sampler, outcomes_dict):
        # Collect measurement outcomes from a CircuitSampler
        operator_in_use = sampler.oplist[0]
        events = sampler.oplist[1].execution_results['counts'].items()

        for key, item in events:
            # Get the binary value corresponding to the outcome
            binary_value = bin(int(key, 16))[2:].zfill(self._num_qubits)

            # Evaluate the operator on the binary value to determine the state
            state = np.real(operator_in_use.eval(binary_value))
            outcomes_dict[state] += item
        print("Outcomes",outcomes_dict)
        return outcomes_dict

    def _redefineBasis(self, Pauli_H):
        # Compute the A state
        ket_A = CircuitStateFn(primitive=self._ansatz).to_matrix()
        ket_A = ket_A.reshape((len(ket_A), 1))
        bra_A = ket_A.conjugate().transpose()

        # Compute the projection of Pauli_H onto the A state
        bra_A_P_ket_A = (bra_A @ Pauli_H.to_matrix() @ ket_A).item()
        P_ket_A = Pauli_H.to_matrix() @ ket_A

        # Define state orthogonal to |A>
        ket_A_ort = (P_ket_A - (bra_A_P_ket_A * ket_A)) / np.sqrt(1 - bra_A_P_ket_A ** 2)
        bra_A_ort = ket_A_ort.conjugate().transpose()

        # Redefine Pauli operators in new basis where |A> and |A_ort> are the basis states
        self._new_sigma_z = PrimitiveOp(ket_A.dot(bra_A) - ket_A_ort.dot(bra_A_ort))
        self._new_sigma_x = PrimitiveOp(ket_A.dot(bra_A_ort) + ket_A_ort.dot(bra_A))
        self._new_sigma_y = PrimitiveOp(-1j * ket_A.dot(bra_A_ort) + 1j * ket_A_ort.dot(bra_A))
        self._new_identity = PrimitiveOp(ket_A.dot(bra_A) + ket_A_ort.dot(bra_A_ort))  # Trivial operator

        return ket_A, ket_A_ort

    def _updateGateArray(self, theta_angle):
        gate_array = {}
        gate_array_derivative = {}

        x_angles = self._x_angles
        num_layers = self._layers

        for i in range(num_layers):
            gate_array[2 * i] = self._new_U(theta_angle, -x_angles[2 * i]).to_matrix()  # U^+
            gate_array[2 * i + 1] = self._new_V(-x_angles[2 * i + 1]).to_matrix()  # V^+
            gate_array[4 * num_layers - 2 * i - 1] = self._new_V(x_angles[2 * i + 1]).to_matrix()  # V
            gate_array[4 * num_layers - 2 * i] = self._new_U(theta_angle, x_angles[2 * i]).to_matrix()  # U

            gate_array_derivative[2 * i] = self._new_U_prime(theta_angle, -x_angles[2 * i]).to_matrix()  # U^+ derivative
            gate_array_derivative[2 * i + 1] = np.zeros((2 * self._num_qubits, 2 * self._num_qubits))  # V^+ derivative
            gate_array_derivative[4 * num_layers - 2 * i - 1] = np.zeros((2 * self._num_qubits, 2 * self._num_qubits))  # V deriv
            gate_array_derivative[4 * num_layers - 2 * i] = self._new_U_prime(theta_angle, x_angles[2 * i]).to_matrix()  # U deriv

        gate_array[2 * num_layers] = self._new_P(theta_angle).to_matrix()
        gate_array_derivative[2 * num_layers] = self._new_P_prime(theta_angle).to_matrix()  # P derivative

        ord_gate_array = collections.OrderedDict(sorted(gate_array.items()))
        ord_gate_array_derivative = collections.OrderedDict(sorted(gate_array_derivative.items()))

        return ord_gate_array, ord_gate_array_derivative

    def _get_bias(self, theta_angle, ket_state):
        bra_state = ket_state.conjugate().transpose()
        GateArrays, _ = self._updateGateArray(theta_angle)
        return (bra_state @ functools.reduce(np.dot, GateArrays.values()) @  ket_state).item()

    def _likelihood(self, d, lambda_i, ket_A):
        delta = np.real(self._get_bias(lambda_i, ket_A))
        return (1. + (-1.)**d * delta) / 2.

    def FischerInfo(self, theta_angle, ket_A):
        GateArray, GateArray_Derivative = self._updateGateArray(theta_angle)

        sum_delta = 0
        sum_delta_p = 0
        x_angles = self._x_angles
        len_angles = len(x_angles)
        L = self._layers

        for i, x in enumerate(x_angles):
            j = i + 1
            if j % 2 == 0:
                t = int(j / 2 - 1)
                # print('even case, j=', j, ' t=', t, ' x=', x)
                C, S, B = self._computeCSBeven(GateArray, self._new_sigma_z, ket_A, t, L)
                Cp, Sp, Bp = self._computeDerCSBeven(GateArray, GateArray_Derivative, self._new_sigma_z, ket_A, t, L)
            else:
                t = int((j - 1) / 2)
                # print('odd case, j=', j, ' t=', t, ' x=', x)
                P = GateArray[2 * L]
                C, S, B = self._computeCSBodd(GateArray, P, ket_A, t, L)
                Cp, Sp, Bp = self._computeDerCSBodd(GateArray, GateArray_Derivative, P, ket_A, t, L)
            # print('C,S,B', C, S, B)
            # print('Cp,Bp', Cp, Bp)

            delta = np.real(C * np.cos(2 * x) + S * np.sin(2 * x) + B)
            delta_p = np.real(Cp * np.cos(2 * x) + Sp * np.sin(2 * x) + Bp)

            sum_delta += delta
            sum_delta_p += delta_p

        sum_delta /= len_angles
        sum_delta_p /= len_angles

        return self._get_fisher_info(sum_delta, sum_delta_p)

    def _compute_likelihood(self, Pauli_H):
        # redefine the basis
        ket_A, _ = self._redefineBasis(Pauli_H)

        # create the binning points for theta
        binning_theta = np.linspace(0, np.pi, self._binning_points)
        # select the binning point for theta using [self._binning_bounds[0]:self._binning_bounds[1]]
        binning_theta = binning_theta[self._binning_bounds[0]:self._binning_bounds[1]]

        # compute likelihood for outcome 0 and 1
        likelihood_0 = [self._likelihood(0, lambda_i, ket_A) for lambda_i in binning_theta]
        likelihood_1 = [self._likelihood(1, lambda_i, ket_A) for lambda_i in binning_theta]

        # compute Fisher information (not being used for now)
        # fisher_information = [self.FischerInfo(lambda_i, ket_A) for lambda_i in binning_theta]

        return likelihood_0, likelihood_1, ket_A

    def _new_P(self, theta):
        # redefinition of Pauli-Z and Pauli-X in the new basis
        return np.cos(theta) * self._new_sigma_z + np.sin(theta) * self._new_sigma_x

    def _new_U(self, theta, x):
        # redefinition of U(theta) gate in the new basis
        identity = PrimitiveOp(np.identity(2 ** self._num_qubits))
        return np.cos(x) * identity - 1j * np.sin(x) * self._new_P(theta)

    def _new_V(self, x):
        # redefinition of V gate in the new basis
        identity = PrimitiveOp(np.identity(2 ** self._num_qubits))
        return np.cos(x) * identity - 1j * np.sin(x) * self._new_sigma_z

    # Derivative redefinition
    def _new_P_prime(self, theta):
        # redefinition of Pauli-Z derivative in the new basis
        return -np.sin(theta) * self._new_sigma_z + np.cos(theta) * self._new_sigma_x

    def _new_U_prime(self, theta, x):
        # redefinition of U(theta) derivative in the new basis
        return -1j * np.sin(x) * self._new_P_prime(theta)

    # *****************************************
    # FISHER-INFORMATION
    # *****************************************
    def _P_ab(self, a, b, L, GateArray):
        if 0 <= a <= b <= 4 * L:
            temp = [GateArray[g] for g in list(range(a, b + 1))]
            if len(temp) > 1:
                return functools.reduce(np.dot, temp)
            else:
                return temp[0]
        else:
            size = int(np.sqrt(GateArray[2 * L].size))
            return np.identity(size)

    def _computeCSBeven(self, GateArr, Z, ket_state, t, L):
        bra_state = ket_state.conjugate().transpose()
        # ket_state = ket_state.to_matrix()
        Z = Z.to_matrix()
        a = self._P_ab(0, 2 * t, L, GateArr)
        b = self._P_ab(2 * t + 2, 4 * L - 2 * t - 2, L, GateArr)
        c = self._P_ab(4 * L - 2 * t, 4 * L, L, GateArr)
        # print('Z',Z)
        # print('a,b,c','\n',a,'\n',b,'\n',c)
        coreC = b - (Z @ b @ Z)  # .to_matrix()
        coreS = (b @ Z) - (b @ Z)  # .to_matrix()
        coreB = b + (Z @ b @ Z)  # .to_matrix()
        # print('c,s,b','\n',coreC,'\n',coreS,'\n',coreB)
        # print('bra',bra_state)
        C = 1 / 2 * (bra_state @ a @ coreC @ c @ ket_state)
        S = -1j / 2 * (bra_state @ a @ coreS @ c @ ket_state)
        B = 1 / 2 * (bra_state @ a @ coreB @ c @ ket_state)
        return C.item(), S.item(), B.item()

    def _computeCSBodd(self, GateArr, P, ket_state, t, L):
        bra_state = ket_state.conjugate().transpose()
        a = self._P_ab(0, 2 * t - 1, L, GateArr)
        b = self._P_ab(2 * t + 1, 4 * L - 2 * t - 1, L, GateArr)
        c = self._P_ab(4 * L - 2 * t + 1, 4 * L, L, GateArr)
        coreC = b - (P @ b @ P)  # CHECK if @ works as .dot()
        coreS = (b @ P) - (b @ P)
        coreB = b + (P @ b @ P)
        C = 1 / 2 * (bra_state @ a @ coreC @ c @ ket_state)
        S = -1j / 2 * (bra_state @ a @ coreS @ c @ ket_state)
        B = 1 / 2 * (bra_state @ a @ coreB @ c @ ket_state)
        return C.item(), S.item(), B.item()

    def _computeDerABCeven(self, GateArr, GateArrDer, t, L):
        size = int(np.sqrt(GateArr[2 * L].size))  # get the size from a matrix
        A1 = self._P_ab(0, 2 * t, L, GateArr)
        B1 = self._P_ab(2 * t + 2, 4 * L - 2 * t - 2, L, GateArr)
        C1 = sum([self._P_ab(4 * L - 2 * t, 4 * L - 2 * k - 1, L, GateArr)
                  @ GateArrDer[4 * L - 2 * k]
                  @ self._P_ab(4 * L - 2 * k + 1, 4 * L, L, GateArr)
                  for k in range(0, t + 1)])

        A2 = A1
        B2 = sum([self._P_ab(2 * t + 2, 4 * L - 2 * k - 1, L, GateArr)
                  @ GateArrDer[4 * L - 2 * k]
                  @ self._P_ab(4 * L - 2 * k + 1, 4 * L - 2 * t - 2, L, GateArr)
                  for k in range(t + 1, L)])
        if isinstance(B2, int):
            if B2 == 0:
                B2 = np.zeros((size, size))
        C2 = self._P_ab(4 * L - 2 * t, 4 * L, L, GateArr)

        A3 = A1
        B3 = self._P_ab(2 * t + 2, 2 * L - 1, L, GateArr)\
            @ GateArrDer[2 * L] \
            @ self._P_ab(2 * L + 1, 4 * L - 2 * t - 2, L, GateArr)
        C3 = C2
        return A1, B1, C1, A2, B2, C2, A3, B3, C3

    def _computeDerABCodd(self, GateArr, GateArrDer, t, L):
        size = int(np.sqrt(GateArr[2 * L].size))  # get the size from a matrix
        A1 = self._P_ab(0, 2 * t - 1, L, GateArr)
        B1 = self._P_ab(2 * t + 1, 4 * L - 2 * t - 1, L, GateArr)
        C1 = sum([self._P_ab(4 * L - 2 * t + 1, 4 * L - 2 * k - 1, L, GateArr)
                  @ GateArrDer[4 * L - 2 * k]
                  @ self._P_ab(4 * L - 2 * k + 1, 4 * L, L, GateArr)
                  for k in range(0, t)])
        if isinstance(C1, int):
            if C1 == 0:
                C1 = np.zeros((size, size))

        A2 = A1
        B2 = B1
        C2 = self._P_ab(4 * L - 2 * t + 1, 4 * L, L, GateArr)

        A3 = A1
        B3 = sum([self._P_ab(2 * t + 1, 4 * L - 2 * k - 1, L, GateArr)
                  @ GateArrDer[4 * L - 2 * k]
                  @ self._P_ab(4 * L - 2 * k + 1, 4 * L - 2 * t - 1, L, GateArr)
                  for k in range(t + 1, L)])
        if isinstance(B3, int):
            if B3 == 0:
                B3 = np.zeros((size, size))
        C3 = C2
        A4 = A1
        B4 = self._P_ab(2 * t + 1, 2 * L - 1, L, GateArr)\
            @ GateArrDer[2 * L] \
            @ self._P_ab(2 * L + 1, 4 * L - 2 * t - 1, L, GateArr)
        C4 = C2
        return A1, B1, C1, A2, B2, C2, A3, B3, C3, A4, B4, C4

    def _computeDerCSBeven(self, GateArr, GateArrDer, Z, ket_state, t, L):
        bra_state = ket_state.conjugate().transpose()
        Z = Z.to_matrix()
        A1, B1, C1, A2, B2, C2, A3, B3, C3 = self._computeDerABCeven(GateArr, GateArrDer, t, L)
        # print('even A1,B1,C1,A2,B2,C2,A3,B3,C3:\n',A1,'\n',B1,'\n',C1,'\n',A2,'\n',B2,'\n',C2,'\n',A3,'\n',B3,'\n',C3)
        C_prime = np.real(bra_state @ A1 @ (B1 - (Z @ B1 @ Z)) @ C1 @ ket_state) + \
            np.real(bra_state @ A2 @ (B2 - (Z @ B2 @ Z)) @ C2 @ ket_state) + \
            1 / 2 * (bra_state @ A3 @ (B3 - (Z @ B3 @ Z)) @ C3 @ ket_state)

        S_prime = np.imag(bra_state @ A1 @ ((B1 @ Z) - (Z @ B1)) @ C1 @ ket_state) + \
            np.imag(bra_state @ A2 @ ((B2 @ Z) - (Z @ B2)) @ C2 @ ket_state) - \
            1j / 2 * (bra_state @ A3 @ ((B3 @ Z) - (Z @ B3)) @ C3 @ ket_state)

        B_prime = np.real(bra_state @ A1 @ (B1 + (Z @ B1 @ Z)) @ C1 @ ket_state) + \
            np.real(bra_state @ A2 @ (B2 + (Z @ B2 @ Z)) @ C2 @ ket_state) + \
            1 / 2 * (bra_state @ A3 @ (B3 + (Z @ B3 @ Z)) @ C3 @ ket_state)
        return C_prime.item(), S_prime.item(), B_prime.item()

    def _computeDerCSBodd(self, GateArr, GateArrDer, P, ket_state, t, L):
        bra_state =  ket_state.conjugate().transpose()
        P_prime = GateArrDer[2 * L]
        A1, B1, C1, A2, B2, C2, A3, B3, C3, A4, B4, C4 = self._computeDerABCodd(GateArr,GateArrDer,t,L)
        #print('odd A1,B1,C1,A2,B2,C2,A3,B3,C3,A4,B4,C4:\n',A1,'\n',B1,'\n',C1,'\n',A2,'\n',B2,'\n',C2,'\n',A3,'\n',B3,'\n',C3,'\n',A4,'\n',B4,'\n',C4)
        C_prime = np.real(bra_state @ A1 @ (B1 - (P @ B1 @ P)) @ C1 @ ket_state) - \
            np.real(bra_state @ A2 @ P @ B2 @ P_prime @ C2 @ ket_state) + \
            np.real(bra_state @ A3 @ (B3 - (P @ B3 @ P)) @ C3 @ ket_state) + \
            1 / 2 * (bra_state @ A4 @ (B4 - (P @ B4 @ P)) @ C4 @ ket_state)
        
        S_prime = np.imag(bra_state @ A1 @ ((B1 @ P) - (P @ B1)) @ C1 @ ket_state) + \
            np.imag(bra_state @ A2 @ B2 @ P_prime @ C2 @ ket_state) + \
            np.imag(bra_state @ A3 @ ((B3 @ P) - (P @ B3)) @ C3 @ ket_state) - \
            1j / 2 * (bra_state @ ( A4 @ ((B4 @ P) - (P @ B4)) @ C4) @ ket_state)
        
        B_prime = np.real(bra_state @ A1 @ (B1 + (P @ B1 @ P)) @ C1 @ ket_state) + \
                np.real(bra_state @ A2 @ P @ B2 @ P_prime @ C2 @ ket_state) + \
                np.real(bra_state @ A3 @ (B3 + (P @ B3 @ P)) @ C3 @ ket_state) + \
                    1/2*(bra_state @ A4 @ (B4 + (P @ B4 @ P)) @ C4 @ ket_state)

        return C_prime.item(), S_prime.item(), B_prime.item()

    def _get_fisher_info(self, delta, delta_p, f = 0.99):
        num = f ** 2 * delta_p ** 2
        den = 1 - f ** 2 * delta ** 2
        return num / den
    