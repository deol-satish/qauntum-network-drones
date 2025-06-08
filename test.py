from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit, Clbit
from utils.ibm_lab_util import build_qc


qr = QuantumRegister(3, name="q")
cr = ClassicalRegister(3, name="c")


teleportation_circuit = build_qc(qr, cr)
s, a, b = qr
c0, c1, c2 = cr
teleportation_circuit.measure(b, c2)
teleportation_circuit.draw("mpl")


import math

teleport_superposition_circuit: QuantumCircuit


# Create a circuit that has the same structure as our teleportation circuit
state_prep = QuantumCircuit(qr, cr)

# Prepare the qubit
state_prep.rx(math.pi / 4, s)

# Put a barrier across all of the wires
state_prep.barrier()
# Add the teleportation circuit to the superposition circuit
teleport_superposition_circuit = state_prep.compose(teleportation_circuit)

teleport_superposition_circuit.draw("mpl", cregbundle=False)




# -------------------------------
# Step 2: Build Noise Model
# -------------------------------
noise_model = NoiseModel()

# Define depolarizing noise
prob_1q = 0.01
prob_2q = 0.02
error_1q = depolarizing_error(prob_1q, 1)
error_2q = depolarizing_error(prob_2q, 2)

# Add depolarizing noise to gates
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rz', 'x', 'z'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# Add thermal relaxation noise
T1 = 50     # μs
T2 = 30     # μs
gate_time = 0.1  # μs
thermal = thermal_relaxation_error(T1, T2, gate_time)
noise_model.add_all_qubit_quantum_error(thermal, ['h', 'x', 'rz'])

# -------------------------------
# Step 3: Simulate with Noise
# -------------------------------
simulator = AerSimulator(noise_model=noise_model)
compiled = transpile(teleportation_circuit, simulator)
result = simulator.run(compiled, shots=1024).result()
counts = result.get_counts()

# -------------------------------
# Step 4: Fidelity Calculation
# -------------------------------
# Get the ideal output state (no noise)
ideal_sim = AerSimulator()
compiled_ideal = transpile(teleportation_circuit, ideal_sim)
ideal_result = ideal_sim.run(compiled_ideal).result()
ideal_state = Statevector.from_int(0, dims=(2, 2))  # |00>

# For actual fidelity comparison, we expect teleportation to yield |ψ> on Bob’s qubit.
# Since our circuit measures both qubits, we simulate the initial prepared state.
prepared_state = Statevector.from_instruction(teleportation_circuit.remove_final_measurements(inplace=False))

# Fidelity between ideal |00> and final prepared state
fidelity = state_fidelity(ideal_state, prepared_state)
print(f"Fidelity of teleportation: {fidelity:.4f}")

# -------------------------------
# Step 5: Latency and Throughput
# -------------------------------
gate_times = {
    'h': 0.1,
    'rz': 0.1,
    'cx': 0.5,
    'x': 0.1,
    'z': 0.1,
    'measure': 0.2
}

# Estimate total latency in microseconds
latency_us = (
    gate_times['h'] +
    gate_times['rz'] +
    gate_times['cx'] +
    gate_times['h'] +
    gate_times['measure'] * 2 +  # measuring two qubits
    gate_times['x'] +
    gate_times['z']
)
print(f"Estimated quantum latency: {latency_us:.2f} μs")

# Define throughput (successful ops/sec) using fidelity threshold
success = 1 if fidelity > 0.9 else 0
throughput = success / (latency_us * 1e-6)  # ops per second
print(f"Estimated quantum throughput: {throughput:.2f} ops/sec")

# -------------------------------
# Step 6: Visualize Results
# -------------------------------
plot_histogram(counts, title="Teleportation with Noise")
plt.show()
