import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

N = 32
ALPHA = 0.1
DT = 0.1
STEPS_PER_FRAME = 50
TOTAL_FRAMES = 3000


def get_acceleration(u, alpha):
    diff = u[1:] - u[:-1]
    linear = diff[1:] - diff[:-1]
    nonlinear = alpha * (diff[1:]**2 - diff[:-1]**2)
    acc = np.zeros_like(u)
    acc[1:-1] = linear + nonlinear
    return acc


def step_verlet(u, v, a, alpha, dt):
    v_half = v + 0.5 * a * dt
    u_new = u + v_half * dt
    a_new = get_acceleration(u_new, alpha)
    v_new = v_half + 0.5 * a_new * dt
    return u_new, v_new, a_new


def get_energies(u, v, modes, freqs_sq):
    u_active = u[1:-1]
    v_active = v[1:-1]
    q_amp = np.dot(u_active, modes)
    q_dot = np.dot(v_active, modes)
    energy = 0.5 * (q_dot**2 + freqs_sq * q_amp**2)
    return energy


def run_simulation():
    main_diag = 2 * np.ones(N - 1)
    off_diag = -1 * np.ones(N - 2)
    matrix_a = (
        np.diag(main_diag) +
        np.diag(off_diag, k=1) +
        np.diag(off_diag, k=-1)
    )
    eigenvals, eigenvecs = np.linalg.eig(matrix_a)

    idx = eigenvals.argsort()
    freqs_sq = eigenvals[idx]
    modes = eigenvecs[:, idx]

    u = np.zeros(N + 1)
    x_axis = np.arange(N + 1)
    u[1:-1] = 2.0 * np.sin(np.pi * x_axis[1:-1] / N)
    v = np.zeros(N + 1)
    acc = get_acceleration(u, ALPHA)

    history_u = []
    history_e = []
    time_axis = []

    for i in range(TOTAL_FRAMES):
        current_time = i * DT * STEPS_PER_FRAME

        for _ in range(STEPS_PER_FRAME):
            u, v, acc = step_verlet(u, v, acc, ALPHA, DT)

        history_u.append(u.copy())
        history_e.append(get_energies(u, v, modes, freqs_sq))
        time_axis.append(current_time)

    return np.array(history_u), np.array(history_e), time_axis, x_axis


def main():
    hist_u, hist_e, time_axis, x_axis = run_simulation()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.3)

    line_particles, = ax1.plot(
        [], [], 'o-', color='royalblue', markersize=5, lw=1
    )
    ax1.set_xlim(0, N)
    ax1.set_ylim(-15, 15)
    ax1.set_ylabel("Displacement")
    ax1.grid(True, alpha=0.3)

    colors = ['red', 'green', 'orange', 'purple', 'black']
    lines_energy = []
    for i in range(5):
        line, = ax2.plot(
            [], [], label=f'Mode {i + 1}', color=colors[i], lw=1.5
        )
        lines_energy.append(line)

    ax2.set_xlim(0, max(time_axis))
    ax2.set_ylim(0, np.max(hist_e[:, :4]) * 1.1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Energy")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    def init():
        line_particles.set_data([], [])
        for line in lines_energy:
            line.set_data([], [])
        return [line_particles] + lines_energy

    def update(frame):
        line_particles.set_data(x_axis, hist_u[frame])

        current_t = time_axis[:frame]
        for i, line in enumerate(lines_energy):
            current_e = hist_e[:frame, i]
            line.set_data(current_t, current_e)

        return [line_particles] + lines_energy

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=TOTAL_FRAMES,
        interval=20,
        blit=True
    )
    anim.save("FPUT_Simulation.gif", fps=30)
    plt.show()


if __name__ == "__main__":
    main()
