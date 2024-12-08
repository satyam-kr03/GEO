import torch
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from scipy.stats import norm
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from torch import nn
from qiskit.primitives import Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch.optim import Adam
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuantumGenerator:
    def __init__(self, num_qubits, shots=10000):
        self.num_qubits = num_qubits
        self.qc = self._create_quantum_circuit()
        self.sampler = Sampler(options={"shots": shots, "seed": algorithm_globals.random_seed})
        self.qnn = self._create_quantum_neural_network()
        self.generator = TorchConnector(self.qnn, algorithm_globals.random.random(self.qc.num_parameters))

    def _create_quantum_circuit(self):
        qc = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))
        ansatz = EfficientSU2(self.num_qubits, reps=5)
        qc.compose(ansatz, inplace=True)
        return qc

    def _create_quantum_neural_network(self):
        return SamplerQNN(
            circuit=self.qc,
            sampler=self.sampler,
            input_params=[],
            weight_params=self.qc.parameters,
            sparse=False,
        )

    def generate_samples(self, batch_size=64, device="cpu"):
        device = torch.device(device)
        samples = []

        for _ in range(batch_size):
            # Generate one sample, take the first 5 elements, and move to the specified device
            sample = self.generator().detach().numpy()[:5]  # Ensure detachment if using PyTorch
            samples.append(sample)

        # Convert the list of samples to a PyTorch tensor
        return torch.tensor(samples, dtype=torch.float32, device=device)

    
    def forward(self):
        return self.generator()  # Generate a quantum-based sample


class ClassicalDiscriminator(nn.Module):
    def __init__(self, input_dim=5):
        super(ClassicalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
