"""Model predictive control for the inverted double pendulum on a cart.

Helper module for ``Robust_Control_Double_Pendulum.ipynb``.

The Euler-Lagrange equations are formulated as an implicit index-1 DAE by
introducing the accelerations ``[ddpos, ddtheta1, ddtheta2]`` as algebraic
states. The MPC objective is energy-based: minimise kinetic energy while
maximising potential energy, which drives swing-up and then stabilisation.

Requires ``do-mpc`` (which pulls in CasADi), not a BayesFlow dependency:

    pip install do-mpc

Adapted from the do-mpc example gallery:
https://www.do-mpc.com/en/latest/example_gallery/DIP.html
"""

import do_mpc
import numpy as np
from casadi import cos, sin, vertcat
from sklearn.cluster import KMeans
from tqdm import tqdm

try:
    from IPython.display import HTML
except ImportError:
    HTML = None

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


def pi_tick_locator_formatter(lo: float, hi: float, max_ticks: int = 4):
    """Return a (MultipleLocator, FuncFormatter) pair for an angle axis.

    Ticks are placed at multiples of π/2, with the step size chosen so that
    at most *max_ticks* ticks appear in [lo, hi].  Labels are formatted as
    fractions/multiples of π (e.g. ``π/2``, ``π``, ``2π``).
    """
    half_pi = np.pi / 2
    span = hi - lo

    step = half_pi
    for k in [1, 2, 3, 4, 6, 8, 12, 16, 24]:
        step = half_pi * k
        if span / step <= max_ticks:
            break

    def _fmt(val, _pos):
        n = round(val / half_pi)
        if n == 0:
            return "$0$"
        if n % 2 == 0:
            m = n // 2
            if m == 1:
                return r"$\pi$"
            if m == -1:
                return r"$-\pi$"
            return rf"${m}\pi$"
        else:
            if n == 1:
                return r"$\pi/2$"
            if n == -1:
                return r"$-\pi/2$"
            return rf"${n}\pi/2$"

    return mticker.MultipleLocator(step), mticker.FuncFormatter(_fmt)


def pick_scenarios(samples: dict[str, np.ndarray], k: int) -> np.ndarray:
    """k-means to pick k representative (m1, m2) vectors. Ordered by increasing distance to mean."""
    samples = np.concatenate([samples["m1"], samples["m2"]], axis=-1)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(samples)
    centres = km.cluster_centers_
    dists = np.linalg.norm(centres - samples.mean(0), axis=1)
    return centres[np.argsort(dists)]


def build_model(
    m0: float,
    l1: float,
    l2: float,
    g: float = 9.81,
) -> do_mpc.model.Model:
    """Build the continuous DAE model of the double inverted pendulum.

    The model uses:
    - States: ``pos``, ``theta`` (2,), ``dpos``, ``dtheta`` (2,)
    - Algebraic states: ``ddpos``, ``ddtheta`` (2,)
    - Input: ``force``
    - Uncertain parameters: ``m1``, ``m2`` (rod masses)
    - Auxiliary expressions: ``E_kin``, ``E_pot``

    Parameters
    ----------
    m0 : float
        Mass of the cart [kg].
    l1, l2 : float
        Full length of the first / second rod [m].  Internally the
        centre-of-mass distances are ``hl1 = l1/2`` and ``hl2 = l2/2``.
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    do_mpc.model.Model
        Configured and set-up model instance.
    """
    hl1 = l1 / 2
    hl2 = l2 / 2

    model = do_mpc.model.Model("continuous")

    # states
    model.set_variable("_x", "pos")
    theta = model.set_variable("_x", "theta", (2, 1))
    dpos = model.set_variable("_x", "dpos")
    dtheta = model.set_variable("_x", "dtheta", (2, 1))

    # algebraic states (accelerations)
    ddpos = model.set_variable("_z", "ddpos")
    ddtheta = model.set_variable("_z", "ddtheta", (2, 1))

    # input
    u = model.set_variable("_u", "force")

    # uncertain parameters (rod masses)
    m1 = model.set_variable("_p", "m1")
    m2 = model.set_variable("_p", "m2")

    # derived symbolic coefficients
    J1 = (m1 * hl1**2) / 3
    J2 = (m2 * hl2**2) / 3

    h1 = m0 + m1 + m2
    h2 = m1 * hl1 + m2 * l1
    h3 = m2 * hl2
    h4 = m1 * hl1**2 + m2 * l1**2 + J1
    h5 = m2 * hl2 * l1
    h6 = m2 * hl2**2 + J2
    h7 = (m1 * hl1 + m2 * l1) * g
    h8 = m2 * hl2 * g

    # ODE right-hand side
    model.set_rhs("pos", dpos)
    model.set_rhs("theta", dtheta)
    model.set_rhs("dpos", ddpos)
    model.set_rhs("dtheta", ddtheta)

    # DAE algebraic equations (Euler-Lagrange = 0)
    euler_lagrange = vertcat(
        # equation 1
        h1 * ddpos
        + h2 * ddtheta[0] * cos(theta[0])
        + h3 * ddtheta[1] * cos(theta[1])
        - (h2 * dtheta[0] ** 2 * sin(theta[0]) + h3 * dtheta[1] ** 2 * sin(theta[1]) + u),
        # equation 2
        h2 * cos(theta[0]) * ddpos
        + h4 * ddtheta[0]
        + h5 * cos(theta[0] - theta[1]) * ddtheta[1]
        - (h7 * sin(theta[0]) - h5 * dtheta[1] ** 2 * sin(theta[0] - theta[1])),
        # equation 3
        h3 * cos(theta[1]) * ddpos
        + h5 * cos(theta[0] - theta[1]) * ddtheta[0]
        + h6 * ddtheta[1]
        - (h5 * dtheta[0] ** 2 * sin(theta[0] - theta[1]) + h8 * sin(theta[1])),
    )
    model.set_alg("euler_lagrange", euler_lagrange)

    # auxiliary expressions (energy)
    E_kin_cart = 0.5 * m0 * dpos**2
    E_kin_p1 = (
        0.5 * m1 * ((dpos + hl1 * dtheta[0] * cos(theta[0])) ** 2 + (hl1 * dtheta[0] * sin(theta[0])) ** 2)
        + 0.5 * J1 * dtheta[0] ** 2
    )
    E_kin_p2 = (
        0.5
        * m2
        * (
            (dpos + l1 * dtheta[0] * cos(theta[0]) + hl2 * dtheta[1] * cos(theta[1])) ** 2
            + (l1 * dtheta[0] * sin(theta[0]) + hl2 * dtheta[1] * sin(theta[1])) ** 2
        )
        + 0.5 * J2 * dtheta[0] ** 2
    )

    E_kin = E_kin_cart + E_kin_p1 + E_kin_p2
    E_pot = m1 * g * hl1 * cos(theta[0]) + m2 * g * (l1 * cos(theta[0]) + hl2 * cos(theta[1]))

    model.set_expression("E_kin", E_kin)
    model.set_expression("E_pot", E_pot)

    model.setup()
    return model


def build_mpc(
    model: do_mpc.model.Model,
    scenarios: np.ndarray,
    n_horizon: int = 100,
    n_robust: int = 1,
    t_step: float = 0.05,
    w_F: float = 0.01,
    F_max: float = 50.0,
    x_bound: float | None = None,
    linear_solver: str = "mumps",
    nlpsol_opts_extra: dict | None = None,
) -> do_mpc.controller.MPC:
    """Build and configure the MPC controller.

    The objective uses an *energy-based* formulation: minimise kinetic
    energy while maximising potential energy (i.e. ``lterm = E_kin - E_pot``).
    An additional input-rate penalty ``w_F`` is applied via ``set_rterm``.

    Parameters
    ----------
    model : do_mpc.model.Model
        The model returned by :meth:`build_model`.
    scenarios : np.ndarray
        Shape ``(n_combinations, 2)``, columns ``[m1, m2]``.  Each row is
        one joint sample; rows are passed directly to the scenario tree
        without Cartesian expansion.
    n_horizon : int
        Prediction horizon length.
    n_robust : int
        Robust horizon (number of scenario branchings).
    t_step : float
        Sampling time [s].
    w_F : float
        Input-rate penalty weight passed to ``mpc.set_rterm(force=w_F)``.
    F_max : float
        Symmetric bound on the input force [N].
    x_bound : float or None
        If given, the cart position ``x`` is constrained to
        ``[-x_bound, x_bound]``.  If None (default), no position
        constraint is imposed.
    linear_solver : str
        IPOPT linear solver, e.g. ``"mumps"`` (default) or ``"ma27"``.
    Returns
    -------
    do_mpc.controller.MPC
    """
    mpc = do_mpc.controller.MPC(model)

    nlp_opts = {
        "ipopt.linear_solver": linear_solver,
        "ipopt.print_level": 0,
        "print_time": 0,
    }
    if nlpsol_opts_extra:
        nlp_opts.update(nlpsol_opts_extra)

    mpc.set_param(
        n_horizon=n_horizon,
        n_robust=n_robust,
        open_loop=0,
        t_step=t_step,
        state_discretization="collocation",
        collocation_type="radau",
        collocation_deg=3,
        collocation_ni=1,
        store_full_solution=True,
        nlpsol_opts=nlp_opts,
    )

    # Energy-based objective: minimise E_kin, maximise E_pot
    mterm = model.aux["E_kin"] - model.aux["E_pot"]
    lterm = model.aux["E_kin"] - model.aux["E_pot"]
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Input-rate penalty
    mpc.set_rterm(force=w_F)

    # Input bounds
    mpc.bounds["lower", "_u", "force"] = -F_max
    mpc.bounds["upper", "_u", "force"] = F_max

    # Optional cart position bounds
    if x_bound is not None:
        mpc.bounds["lower", "_x", "pos"] = -x_bound
        mpc.bounds["upper", "_x", "pos"] = x_bound

    # Scenario tree: one combination per row, no Cartesian expansion
    n_combinations = scenarios.shape[0]
    p_template = mpc.get_p_template(n_combinations)
    for i in range(n_combinations):
        p_template["_p", i, "m1"] = scenarios[i, 0]
        p_template["_p", i, "m2"] = scenarios[i, 1]

    def p_fun(t_now):  # noqa: ARG001
        return p_template

    mpc.set_p_fun(p_fun)
    mpc._p_template = p_template  # expose for in-place scenario updates

    mpc.setup()

    return mpc


def build_simulator(
    model: do_mpc.model.Model,
    m1_true: float,
    m2_true: float,
    t_step: float = 0.05,
) -> do_mpc.simulator.Simulator:
    """Build a simulator for the double inverted pendulum.

    Parameters
    ----------
    model : do_mpc.model.Model
    m1_true, m2_true : float
        True rod masses used during simulation [kg].
    t_step : float
        Integration step size [s].

    Returns
    -------
    do_mpc.simulator.Simulator
    """
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(
        integration_tool="idas",
        abstol=1e-8,
        reltol=1e-8,
        t_step=t_step,
    )

    # Constant true parameters
    p_template = simulator.get_p_template()
    p_template["m1"] = m1_true
    p_template["m2"] = m2_true

    def p_fun(t_now):  # noqa: ARG001
        return p_template

    simulator.set_p_fun(p_fun)
    simulator.setup()
    return simulator


def build_estimator(
    model: do_mpc.model.Model,
) -> do_mpc.estimator.StateFeedback:
    """Build a simple state-feedback estimator (perfect measurements).

    Parameters
    ----------
    model : do_mpc.model.Model

    Returns
    -------
    do_mpc.estimator.StateFeedback
    """
    estimator = do_mpc.estimator.StateFeedback(model)
    return estimator


def run_closed_loop(
    mpc: do_mpc.controller.MPC,
    simulator: do_mpc.simulator.Simulator,
    estimator: do_mpc.estimator.StateFeedback,
    x0: np.ndarray,
    n_steps: int,
) -> dict:
    """Run a closed-loop MPC simulation.

    Parameters
    ----------
    mpc : do_mpc.controller.MPC
    simulator : do_mpc.simulator.Simulator
    estimator : do_mpc.estimator.StateFeedback
    x0 : np.ndarray
        Initial state vector, shape ``(6, 1)`` or ``(6,)``.
        Order: ``[pos, theta1, theta2, dpos, dtheta1, dtheta2]`` (do-mpc order).
    n_steps : int
        Number of closed-loop steps to simulate.

    Returns
    -------
    dict
        ``{'mpc': ..., 'simulator': ..., 'estimator': ...,
          'x_traj': np.ndarray, 'u_traj': np.ndarray, 't': np.ndarray,
          'success': bool, 'n_steps_completed': int, 'error': str | None}``

        ``x_traj`` has columns in do-mpc order:
        ``[pos, theta1, theta2, dpos, dtheta1, dtheta2]``.
        ``u_traj`` has shape ``(n_completed, 1)``.
        ``t`` is the time vector (may be shorter than n_steps if failed).
        ``success`` is False when the integrator or solver raised an error.
        ``error`` contains the error message string if ``success`` is False.
    """
    x0 = np.asarray(x0).reshape(-1, 1)

    # Reset data stores so components can be reused across calls
    mpc.data.init_storage()
    simulator.data.init_storage()
    estimator.data.init_storage()

    # Set initial state for all components
    simulator.x0 = x0
    mpc.x0 = x0
    estimator.x0 = x0
    mpc.set_initial_guess()

    success = True
    error_msg = None

    # Closed-loop iterations
    for step in tqdm(range(n_steps)):
        try:
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"\nClosed-loop failed at step {step}/{n_steps}: {type(e).__name__}")
            break

    # Extract trajectory arrays
    x_traj = np.array(mpc.data["_x"])
    raw_u = np.array(mpc.data["_u"])
    raw_t = np.array(mpc.data["_time"]).ravel()
    n_steps_done = len(raw_t)

    # Pre-extract prediction data to numpy so the result dict is self-contained
    # (no live mpc object needed for animation; enables component reuse and pickling)
    _pred_keys = [("_x", "theta", 0), ("_x", "theta", 1), ("_u", "force", 0)]
    predictions = {
        key: np.concatenate(
            [mpc.data.prediction(key, t_ind=i) for i in range(n_steps_done)], axis=0
        )  # (n_steps, n_horizon_pts, n_scenarios)
        for key in _pred_keys
    }

    return {
        "mpc": mpc,
        "simulator": simulator,
        "estimator": estimator,
        "x_traj": x_traj,
        "u_traj": raw_u,
        "t": raw_t,
        "predictions": predictions,
        "success": success,
        "n_steps_completed": n_steps_done,
        "error": error_msg,
    }


def animate_with_predictions(
    data: dict,
    l1: float,
    l2: float,
    skip: int = 1,
    filename: str | None = None,
    color: str = "black",
    color_scenarios: str = "#132a70",
    frame: int | None = None,
    legend_fontsize: int = 12,
    legend_location: str = "lower right",
    dpi: float | None = None,
) -> "HTML | None":
    """Animate a closed-loop run with MPC scenario-tree predictions.

    Closed-loop history is drawn as solid lines; predictions as faint
    lines — one branch per robust scenario.

    Parameters
    ----------
    data : dict
        Result dict from :meth:`run_closed_loop`.
    l1, l2 : float
        Full rod lengths [m].
    skip : int
        Render every *skip*-th frame.
    filename : str or None
        If given, save the animation instead of returning it. A ``.gif`` suffix
        selects the Pillow writer; any other suffix uses ``ffmpeg``.
    color : str
        Color for the closed-loop measured trajectory. Default ``"black"``.
    color_scenarios : str
        Color for the MPC scenario prediction fan. Default ``"#132a70"``.

    Returns
    -------
    IPython.display.HTML or None
    """
    run = data
    n_scenarios = run["predictions"][("_x", "theta", 0)].shape[2]

    xt = run["x_traj"]
    cx = xt[:, 0]
    t1x = cx + l1 * np.sin(xt[:, 1])
    t1y = l1 * np.cos(xt[:, 1])
    t2x = t1x + l2 * np.sin(xt[:, 2])
    t2y = t1y + l2 * np.cos(xt[:, 2])

    t_all = run["t"]
    t_step = float(t_all[1] - t_all[0]) if len(t_all) > 1 else 0.04
    frame_indices = np.arange(0, len(t_all), skip)

    max_extent = l1 + l2
    x_range = max(np.abs(cx).max() + max_extent + 0.3, max_extent + 0.5)
    y_extent = max_extent + 0.3
    all_th = np.concatenate([xt[:, 1], xt[:, 2]])
    th_lo = min(all_th.min(), 0.0) - 0.3
    th_hi = max(all_th.max(), np.pi) + 0.3
    u_lim = max(np.abs(run["u_traj"]).max() * 1.1, 1.0)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.2, 1],
        hspace=0.35,
        wspace=0.30,
        left=0.05,
        right=0.97,
        top=0.93,
        bottom=0.10,
    )
    ax_cart = fig.add_subplot(gs[:, 0])
    ax_phi = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])

    ax_cart.set_aspect("equal", adjustable="datalim")
    ax_cart.set_xlim(-x_range, x_range)
    ax_cart.set_ylim(-y_extent, y_extent)
    ax_cart.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax_cart.set_title("Double Pendulum on Cart")
    ax_cart.set_xlabel("x [m]")
    ax_cart.tick_params(left=False, labelleft=False)

    ax_phi.set_xlim(t_all[0], t_all[-1])
    ax_phi.set_ylim(th_lo, th_hi)
    _loc, _fmt = pi_tick_locator_formatter(th_lo, th_hi)
    ax_phi.yaxis.set_major_locator(_loc)
    ax_phi.yaxis.set_major_formatter(_fmt)
    ax_phi.tick_params(bottom=True, labelbottom=False)
    ax_phi.set_ylabel("Angle [rad]")
    ax_phi.set_title("State")

    ax_u.set_xlim(t_all[0], t_all[-1])
    ax_u.set_ylim(-u_lim, u_lim)
    ax_u.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, symmetric=True))
    ax_u.axhline(0, color="gray", linewidth=0.5)
    ax_u.set_xlabel("Time [s]")
    ax_u.set_ylabel("F [N]")
    ax_u.set_title("Force")

    pred_alpha = 0.5
    (pend_line,) = ax_cart.plot([], [], "-o", color=color, markersize=4, linewidth=1.5)
    (cart_marker,) = ax_cart.plot([], [], "s", color=color, markersize=10)
    (th1_hist,) = ax_phi.plot([], [], "-", color=color, linewidth=3.5)
    (th2_hist,) = ax_phi.plot([], [], "--", color=color, linewidth=3.5)
    (u_hist,) = ax_u.plot([], [], "-", color=color, linewidth=3.5)
    th1_pred = [
        ax_phi.plot([], [], "-", color=color_scenarios, alpha=pred_alpha, linewidth=2)[0] for _ in range(n_scenarios)
    ]
    th2_pred = [
        ax_phi.plot([], [], "--", color=color_scenarios, alpha=pred_alpha, linewidth=2)[0] for _ in range(n_scenarios)
    ]
    u_pred = [
        ax_u.plot([], [], "-", color=color_scenarios, alpha=pred_alpha, linewidth=2)[0] for _ in range(n_scenarios)
    ]

    ax_phi.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=1.5, label="Measurement"),
            Line2D([0], [0], color=color_scenarios, alpha=pred_alpha, linewidth=1.5, label="Prediction"),
            Line2D([0], [0], color="black", linestyle="-", label=r"$\phi_1$"),
            Line2D([0], [0], color="black", linestyle="--", label=r"$\phi_2$"),
        ],
        fontsize=legend_fontsize,
        loc=legend_location,
        framealpha=0.8,
    )
    ax_u.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=1.5, label="Applied"),
            Line2D([0], [0], color=color_scenarios, alpha=pred_alpha, linewidth=1.5, label="Plan"),
        ],
        fontsize=legend_fontsize,
        loc=legend_location,
        framealpha=0.8,
    )

    all_artists = [pend_line, cart_marker, th1_hist, th2_hist, u_hist] + th1_pred + th2_pred + u_pred

    def init():
        for art in all_artists:
            art.set_data([], [])
        return all_artists

    def update(frame):
        fidx = min(int(frame), len(t_all) - 1)
        t_now = t_all[fidx]
        pidx = min(fidx, len(cx) - 1)
        cart_marker.set_data([cx[pidx]], [0])
        pend_line.set_data([cx[pidx], t1x[pidx], t2x[pidx]], [0, t1y[pidx], t2y[pidx]])
        xi = np.arange(0, min(fidx + 1, len(xt)))
        th1_hist.set_data(t_all[xi], xt[xi, 1])
        th2_hist.set_data(t_all[xi], xt[xi, 2])
        ui = np.arange(0, min(fidx + 1, len(run["u_traj"])))
        u_hist.set_data(t_all[ui], run["u_traj"][ui, 0])
        tidx = min(fidx, len(run["t"]) - 1)
        p1 = run["predictions"][("_x", "theta", 0)][tidx]
        p2 = run["predictions"][("_x", "theta", 1)][tidx]
        pu = run["predictions"][("_u", "force", 0)][tidx]
        t_px = t_now + np.arange(p1.shape[0]) * t_step
        t_pu = t_now + np.arange(pu.shape[0]) * t_step
        for s in range(n_scenarios):
            th1_pred[s].set_data(t_px, p1[:, s])
            th2_pred[s].set_data(t_px, p2[:, s])
            u_pred[s].set_data(t_pu, pu[:, s])
        return all_artists

    if frame is not None:
        init()
        update(frame)
        if filename:
            fig.savefig(filename)
            plt.close(fig)
            return None
        return fig

    anim = FuncAnimation(fig, update, frames=frame_indices, init_func=init, interval=t_step * skip * 1000, blit=True)

    if filename:
        # ffmpeg quantises GIFs per frame without a shared palette; Pillow is cleaner and smaller
        writer = PillowWriter(fps=1.0 / (t_step * skip)) if filename.endswith(".gif") else "ffmpeg"
        anim.save(filename, writer=writer, dpi=dpi)
        plt.close(fig)
        return None
    plt.close(fig)
    if HTML is not None:
        return HTML(anim.to_html5_video())
    plt.show()
    return None
