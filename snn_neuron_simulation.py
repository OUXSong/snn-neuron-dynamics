import numpy as np
import matplotlib.pyplot as plt


class LIFNeuron:
    def __init__(
        self,
        v_rest=-65e-3,
        v_reset=-65e-3,
        v_th=-50e-3,
        tau_m=20e-3,
        R=10e6,
        dt=1e-3,
    ):
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.tau_m = tau_m
        self.R = R
        self.dt = dt

    def step(self, v_prev, current):
        """Advance one Euler step for membrane potential."""
        dv_dt = -(v_prev - self.v_rest) / self.tau_m + (self.R * current) / self.tau_m
        v_next = v_prev + dv_dt * self.dt

        if v_next >= self.v_th:
            return self.v_reset, True
        return v_next, False


def generate_pulse_current(T=1.0, dt=1e-3, amplitude=2e-9, freq=25):
    """Generate a train of input current pulses."""
    time = np.arange(0, T, dt)
    current = np.zeros_like(time)
    pulse_times = np.arange(0, T, 1 / freq)

    for t in pulse_times:
        idx = int(t / dt)
        if idx < len(current):
            current[idx] = amplitude

    return time, current


def simulate_neuron(T=1.0, dt=1e-3, amplitude=2e-9, freq=25):
    """Simulate a single LIF neuron and return time, voltage, spikes."""
    time = np.arange(0, T, dt)
    v = np.full_like(time, -65e-3, dtype=float)
    spikes = np.zeros_like(time, dtype=bool)
    current = np.zeros_like(time)

    pulse_times = np.arange(0, T, 1 / freq)
    for t in pulse_times:
        idx = int(t / dt)
        if idx < len(current):
            current[idx] = amplitude

    model = LIFNeuron(dt=dt)

    for i in range(1, len(time)):
        v[i], fired = model.step(v[i - 1], current[i])
        if fired:
            spikes[i] = True

    return time, v, current, spikes


def plot_results(time, v, current, spikes):
    """Visualize neuron dynamics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(time, current, color='tab:blue', lw=1.5)
    axes[0].set_ylabel('Input current (A)')
    axes[0].set_title('LIF neuron dynamics')
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, v, color='tab:orange', lw=2, label='Membrane potential')
    axes[1].scatter(time[spikes], v[spikes], color='tab:red', s=25, label='Spike')
    axes[1].axhline(-50e-3, color='black', linestyle='--', lw=1, label='Threshold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    T = 1.0
    dt = 1e-3
    amplitude = 2e-9
    freq = 25

    time, v, current, spikes = simulate_neuron(T=T, dt=dt, amplitude=amplitude, freq=freq)
    plot_results(time, v, current, spikes)
    print(f"Total spikes: {spikes.sum()}")
    print(f"Spike times (s): {time[spikes][:10]}")
