# Van Van Chau(1), Hao Anh Nguyen(1), Tuan Minh Nguyen(1,*), Chinda Thorn(2), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Viet Nam, vancv@ptit.edu.vn, haona@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Bangkok, Thailand, s6807011866029@email.kmutnb.ac.th, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding authors
#
"""Classical CAE and QCAE (QAOA-inspired ansatz) models.

The quantum layer is simulated by explicit statevector propagation
implemented with PyTorch complex tensors, rather than through Qiskit. This
makes the whole quantum-classical hybrid natively autograd-differentiable
(gradients flow through the statevector simulation directly), which is a
standard simulator-only alternative to the parameter-shift rule (PSR) used
by Kit et al. (2025) for real-hardware compatibility. Because this study
only targets simulation (no real quantum device), backpropagation through
the simulated statevector is used instead of PSR; this is a deliberate
scope reduction, stated explicitly as a limitation in the paper.

Gate set (H, RY, RZ, CNOT) and the angle-encoding scheme follow Eqs. (1)-(7)
and the ansatz structure in Fig. 5 of Kit et al. (2025).
"""
import torch
import torch.nn as nn

N_QUBITS = 4
DIM = 2 ** N_QUBITS


def _batched_kron(a, b):
    # a: (batch, m, m), b: (batch, n, n) -> (batch, m*n, m*n)
    batch, m, _ = a.shape
    _, n, _ = b.shape
    out = torch.einsum("bij,bkl->bikjl", a, b)
    return out.reshape(batch, m * n, n * m if False else m * n, ) if False else out.reshape(batch, m * n, m * n)


def _identity_batch(batch, size, device, dtype):
    eye = torch.eye(size, device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(batch, size, size)


def _embed_single_qubit(gate_batch, qubit_idx, n_qubits, device):
    # gate_batch: (batch, 2, 2) complex
    batch = gate_batch.shape[0]
    dtype = gate_batch.dtype
    mat = None
    for q in range(n_qubits):
        g = gate_batch if q == qubit_idx else _identity_batch(batch, 2, device, dtype)
        mat = g if mat is None else _batched_kron(mat, g)
    return mat


def _ry_batch(theta):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    zero = torch.zeros_like(theta)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    mat = torch.stack([row0, row1], dim=-2)
    return mat.to(torch.complex64)


def _rz_batch(phi):
    e_m = torch.exp(-1j * phi.to(torch.complex64) / 2)
    e_p = torch.exp(1j * phi.to(torch.complex64) / 2)
    zero = torch.zeros_like(e_m)
    row0 = torch.stack([e_m, zero], dim=-1)
    row1 = torch.stack([zero, e_p], dim=-1)
    mat = torch.stack([row0, row1], dim=-2)
    return mat


def _cnot_matrix(control, target, n_qubits, device):
    dim = 2 ** n_qubits
    mat = torch.zeros(dim, dim, dtype=torch.complex64, device=device)
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 1:
            bits[target] ^= 1
        j = 0
        for b in bits:
            j = (j << 1) | b
        mat[j, i] = 1.0
    return mat


class QAOAInspiredLayer(nn.Module):
    """QAOA-inspired variational ansatz: angle-encode input, then p layers of
    (ring CNOT entanglement + trainable RY/RZ), returning <Z_i> expectation
    values for each qubit as the quantum latent vector."""

    def __init__(self, n_qubits=N_QUBITS, p_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.p_layers = p_layers
        self.theta = nn.Parameter(torch.randn(p_layers, n_qubits) * 0.1)
        self.phi = nn.Parameter(torch.randn(p_layers, n_qubits) * 0.1)
        cnots = []
        for l in range(p_layers):
            layer_cnots = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
            cnots.append(layer_cnots)
        self._cnot_pairs = cnots
        self._cnot_cache = {}

    def _get_cnot(self, control, target, device):
        key = (control, target, device)
        if key not in self._cnot_cache:
            self._cnot_cache[key] = _cnot_matrix(control, target, self.n_qubits, device)
        return self._cnot_cache[key]

    def forward(self, y):
        # y: (batch, n_qubits) real-valued classical compressed features
        batch = y.shape[0]
        device = y.device
        dim = 2 ** self.n_qubits
        state = torch.zeros(batch, dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0

        # angle encoding: RY(y_i) on qubit i
        for q in range(self.n_qubits):
            gate = _ry_batch(y[:, q])
            full = _embed_single_qubit(gate, q, self.n_qubits, device)
            state = torch.bmm(full, state.unsqueeze(-1)).squeeze(-1)

        # variational layers
        for l in range(self.p_layers):
            for (c, t) in self._cnot_pairs[l]:
                cnot = self._get_cnot(c, t, device)
                state = torch.matmul(cnot, state.unsqueeze(-1)).squeeze(-1)
            for q in range(self.n_qubits):
                theta_q = self.theta[l, q].expand(batch)
                gate = _ry_batch(theta_q)
                full = _embed_single_qubit(gate, q, self.n_qubits, device)
                state = torch.bmm(full, state.unsqueeze(-1)).squeeze(-1)
            for q in range(self.n_qubits):
                phi_q = self.phi[l, q].expand(batch)
                gate = _rz_batch(phi_q)
                full = _embed_single_qubit(gate, q, self.n_qubits, device)
                state = torch.bmm(full, state.unsqueeze(-1)).squeeze(-1)

        probs = (state.real ** 2 + state.imag ** 2)  # (batch, dim)
        expZ = torch.zeros(batch, self.n_qubits, device=device)
        for q in range(self.n_qubits):
            signs = torch.ones(dim, device=device)
            for i in range(dim):
                bit = (i >> (self.n_qubits - 1 - q)) & 1
                signs[i] = 1.0 if bit == 0 else -1.0
            expZ[:, q] = (probs * signs.unsqueeze(0)).sum(dim=1)
        return expZ


class ClassicalEncoder(nn.Module):
    def __init__(self, latent_dim=N_QUBITS, hidden_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, stride=2, padding=1)   # 28->14
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1)  # 14->7
        self.act = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(hidden_channels * 2 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = h.flatten(1)
        return self.fc(h)


class ClassicalDecoder(nn.Module):
    def __init__(self, latent_dim=N_QUBITS, hidden_channels=8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.fc = nn.Linear(latent_dim, hidden_channels * 2 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 3, stride=2, padding=1, output_padding=1)  # 7->14
        self.deconv2 = nn.ConvTranspose2d(hidden_channels, 1, 3, stride=2, padding=1, output_padding=1)  # 14->28
        self.act = nn.LeakyReLU(0.1)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.hidden_channels * 2, 7, 7)
        h = self.act(self.deconv1(h))
        return torch.sigmoid(self.deconv2(h))


class ClassicalCAE(nn.Module):
    def __init__(self, latent_dim=N_QUBITS, hidden_channels=8):
        super().__init__()
        self.encoder = ClassicalEncoder(latent_dim, hidden_channels)
        self.decoder = ClassicalDecoder(latent_dim, hidden_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class QCAE(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, p_layers=2, hidden_channels=8):
        super().__init__()
        self.encoder = ClassicalEncoder(n_qubits, hidden_channels)
        self.pre_angle_norm = nn.BatchNorm1d(n_qubits)
        self.quantum = QAOAInspiredLayer(n_qubits, p_layers)
        self.decoder = ClassicalDecoder(n_qubits, hidden_channels)

    def forward(self, x):
        y = self.encoder(x)
        # BatchNorm keeps the pre-tanh distribution at unit scale regardless of
        # how large the raw encoder output grows under training; without it,
        # large learning rates drive the encoder output to a magnitude where
        # tanh saturates to +-1 for every sample, collapsing the angle-encoded
        # state (and hence the reconstruction) to be independent of the input.
        y = self.pre_angle_norm(y)
        y = torch.tanh(y) * torch.pi
        z = self.quantum(y)
        return self.decoder(z)


if __name__ == "__main__":
    x = torch.rand(4, 1, 28, 28)
    cae = ClassicalCAE()
    print("CAE out", cae(x).shape)
    qcae = QCAE(p_layers=2)
    out = qcae(x)
    print("QCAE out", out.shape, out.min().item(), out.max().item())
    n_params_q = sum(p.numel() for p in qcae.quantum.parameters())
    print("quantum trainable params:", n_params_q)
