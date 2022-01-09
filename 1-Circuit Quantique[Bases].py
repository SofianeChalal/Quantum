# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 13:27:45 2022

@author: User1
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# Aer's qasm_simulator
simulator = QasmSimulator()

# Création d'un circuit Quantique
circuit = QuantumCircuit(2, 2)

# Activation de la porte d'Hadamard au qubit 0
circuit.h(0)

# Porte CNOT (CX) qubit  de controle 0 et qubit 1 comme cible
circuit.cx(0, 1)

# Measure des résultat
circuit.measure([0,1], [0,1])


compiled_circuit = transpile(circuit, simulator)

# Execution du circuit.
job = simulator.run(compiled_circuit, shots=1000)

# Saisir les résultats du travail
result = job.result()

# Comptage
counts = result.get_counts(compiled_circuit)
print("\nCompte totale pour 00 et 11  : ",counts)

# Draw the circuit
circuit.draw()

# Tracé de l'histograme
plot_histogram(counts)