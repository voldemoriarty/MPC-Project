
from typing import Callable
from given6.homework import problem
import numpy as np 
import matplotlib.pyplot as plt


import os 
WORKING_DIR = os.path.split(__file__)[0]

def lqr_factor_step(N: int, nl: problem.NewtonLagrangeQP) -> problem.NewtonLagrangeFactors:

    Pk = nl.QN
    sk = nl.qN

    K_all = []
    s_all = [sk]
    P_all = [Pk]
    e_all = []

    for k in range(N-1,-1,-1):
        R = nl.Rk[k] + nl.Bk[k].T @ Pk @ nl.Bk[k]
        S = nl.Sk[k] + nl.Bk[k].T @ Pk @ nl.Ak[k]
        y = Pk @ nl.ck[k] + sk 

        Kk = -np.linalg.solve(R,S)
        ek = -np.linalg.solve(R,nl.Bk[k].T@y + nl.rk[k])
        sk = S.T@ek + nl.Ak[k].T@y + nl.qk[k]
        Pk = nl.Qk[k] + nl.Ak[k].T @ Pk @ nl.Ak[k] + S.T @ Kk

        K_all = [Kk] + K_all
        s_all = [sk] + s_all
        P_all = [Pk] + P_all
        e_all = [ek] + e_all
        

    return problem.NewtonLagrangeFactors(K_all, s_all, P_all, e_all)

def symmetric(P):
    return 0.5 * (P.T + P)

def lqr_solve_step(
    prob: problem.Problem,
    nl: problem.NewtonLagrangeQP,
    fac: problem.NewtonLagrangeFactors
) -> problem.NewtonLagrangeUpdate: 
    dx = [np.zeros((prob.ns, 1))]
    du = []
    p = []

    for k in range(prob.N):
        du_k = fac.K[k] @ dx[k] + fac.e[k]
        dx_k = nl.Ak[k] @ dx[k] + nl.Bk[k] @ du_k + nl.ck[k]
        pk = fac.P[k+1] @ dx_k + fac.s[k+1]

        dx.append(dx_k)
        du.append(du_k)
        p.append(pk)

    return problem.NewtonLagrangeUpdate(np.array(dx).reshape([prob.N+1,prob.ns]), \
                                        np.array(du).reshape([prob.N, prob.nu]), \
                                        np.array(p).reshape([prob.N,-1]))


def armijo_condition(merit: problem.FullCostFunction, x_plus, u_plus, x, u, dx, du, c, σ, α):
    φ, g, dJdx, dJdu = merit.phi, merit.h, merit.dJdx, merit.dJdu
    return φ(c, x_plus, u_plus) <= φ(c, x, u) + σ * α * (dJdx(x, u) @ dx + dJdu(x,u)@du - c * g(x,u))


def armijo_linesearch(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, merit: problem.FullCostFunction, *, σ=1e-4) -> problem.NLIterate:
    alpha = 1.0
    c = 10.0 * update.p.max()

    pplus = update.p 
    uplus = zk.u + alpha*update.du 
    xplus = zk.x + alpha*update.dx 

    v = problem.to_vec

    while not armijo_condition(merit, v(xplus), v(uplus), v(zk.x), v(zk.u), v(update.dx), v(update.du), c, σ, alpha):
        alpha = 0.8 * alpha
        
        uplus = zk.u + alpha*update.du 
        xplus = zk.x + alpha*update.dx 
    
    return problem.NLIterate(xplus, uplus, pplus)


def update_iterate(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, *, linesearch: bool, merit_function: problem.FullCostFunction=None) -> problem.NLIterate:
    """Take the current iterate zk and the Newton-Lagrange update and return a new iterate. 

    If linesearch is True, then also perform a linesearch procedure. 

    Args:
        zk (problem.NLIterate): Current iterate 
        update (problem.NewtonLagrangeUpdate): Newton-Lagrange step 
        linesearch (bool): Perform line search or not? 
        merit_function (problem.FullCostFunction, optional): The merit function used for linesearch. Defaults to None.

    Raises:
        ValueError: If no merit function was passed, but linesearch was requested. 

    Returns:
        problem.NLIterate: Next Newton-Lagrange iterate.
    """
    
    if linesearch:
        if merit_function is None:
            raise ValueError("No merit function was passed but line search was requested")
        return armijo_linesearch(zk, update, merit_function)
    
    # Hint: The initial state must remain fixed. Only update from time index 1!
    xnext = zk.x + update.dx
    unext = zk.u + update.du
    pnext = update.p

    return problem.NLIterate(
        x = xnext,
        u = unext, 
        p = pnext 
    )


def is_posdef(M):
    return np.min(np.linalg.eigvalsh(M)) > 0

def check_qp(qp: problem.NewtonLagrangeQP):
    for k in range(len(qp.Qk)):
        if not is_posdef(np.vstack([
            np.hstack([qp.Qk[k], qp.Sk[k].T]), 
            np.hstack([qp.Sk[k], qp.Rk[k]])
        ])):
            return False
    return is_posdef(qp.QN)

def regularize(qp: problem.NewtonLagrangeQP):
    """Regularize the problem.

    If the given QP (obtained as a linearization of the problem) is nonconvex, 
    add an increasing multiple of the identity to the Hessian 
    until it is positive definite. 

    Side effects: the passed qp is modified by the regularization!

    Args:
        qp (problem.NewtonLagrangeQP): Linearization of the optimal control problem
    """
    lamda = 1e-6

    while not check_qp(qp):
        for k in range(len(qp.Qk)):
            qp.Qk[k,...] += lamda*np.eye(qp.Qk.shape[-1])
            qp.Rk[k,...] += lamda*np.eye(qp.Rk.shape[-1])
        
        lamda *= 2.0


def newton_lagrange(p: problem.Problem,
         initial_guess = problem.NLIterate, cfg: problem.NewtonLagrangeCfg = None, *,
         log_callback: Callable = lambda *args, **kwargs: ...
) -> problem.NewtonLagrangeStats:
    """Newton Lagrange method for nonlinear OCPs
    Args:
        p (problem.Problem): The problem description 
        initial_guess (NLIterate, optional): Initial guess. Defaults to problem.NewtonLagrangeIterate.
        cfg (problem.NewtonLagrangeCfg, optional): Settings. Defaults to None.
        log_callback (Callable): A function that takes the iteration count and the current iterate. Useful for logging purposes. 

    Returns:
        Solver stats  
    """
    stats = problem.NewtonLagrangeStats(0, initial_guess)
    # Set the default config if None was passed 
    if cfg is None:
        cfg = problem.NewtonLagrangeCfg()

    # Get the merit function ingredients in case line search was requested 
    if cfg.linesearch:
        full_cost = problem.build_cost_and_constraint(p)
    else: 
        full_cost = None # We don't need it in this case 
    
    QP_sym = problem.construct_newton_lagrange_qp(p)
    zk = initial_guess

    for it in range(cfg.max_iter):
        qp_it = QP_sym(zk)

        if cfg.regularize:
            regularize(qp_it)

        factor = lqr_factor_step(p.N, qp_it)

        update = lqr_solve_step(p, qp_it, factor)

        zk = update_iterate(zk, update, linesearch=cfg.linesearch, merit_function=full_cost)

        stats.n_its = it 
        stats.solution = zk 
        # Call the logger. 
        log_callback(stats)

        # Sloppy heuristics as termination criteria.
        # In a real application, it's better to check the violation of the KKT conditions.
        # e.g., terminate based on the norm of the gradients of the Lagrangian.
        if np.linalg.norm(update.du.squeeze(), ord=np.inf)/np.linalg.norm(zk.u) < 1e-4:
            stats.exit_message = "Converged"
            stats.success = True 
            return stats

        elif np.any(np.linalg.norm(update.du) > 1e4): 
            stats.exit_message = "Diverged"
            return stats
        
    stats.exit_message = "Maximum number of iterations exceeded"
    return stats


def exercise1():
    print("Assignment 6.1.")
    p = problem.Problem()
    qp = problem.construct_newton_lagrange_qp(p)

def fw_euler(f, Ts):
    return lambda x,u,t: x + Ts*f(x,u)

def test_linear_system():

    p = problem.ToyProblem(cont_dynamics = problem.LinearSystem(), N=100)
    u0 = np.zeros((p.N, p.nu))
    x0 = np.zeros((p.N+1, p.ns))
    x0[0] = p.x0
    initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))

    logger = problem.Logger(p, initial_guess)
    result = newton_lagrange(p, initial_guess, log_callback=logger)
    assert result.success, "Newton Lagrange did not converge on a linear system! Something is wrong!"
    assert result.n_its < 2, "Newton Lagrange took more than 2 iterations!"

def exercise2():
    print("Assignment 6.2.")
    from rcracers.simulator.core import simulate
    
    # Build the problem 
    p = problem.ToyProblem()
    # policy = lambda y,t,log: np.zeros([p.nu,1])
    # dynamics = fw_euler(p.cont_dynamics, p.Ts)

    # Select initial guess by running an open-loop simulation
    # x = simulate(p.x0.reshape([-1,1]), dynamics, p.N, policy=policy)
    x = np.zeros([p.N+1,p.ns])
    u = np.zeros([p.N,p.nu])
    _p = np.zeros([p.N,p.ns])
    x[0,:] = p.x0
    initial_guess = problem.NLIterate(x, u, _p)
    
    logger = problem.Logger(p, initial_guess)
    stats = newton_lagrange(p, initial_guess, log_callback=logger)
    from given6.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "figs/Assignment6-2"))

    plt.figure()
    plt.plot(stats.solution.x)
    plt.plot(stats.solution.u)
    plt.grid()
    plt.legend(["x_1", "x_2", "x_3", "u"])
    plt.xlabel("Time Step")
    plt.ylabel("State Trajectory")
    plt.title("State Evolution with Time")
    plt.savefig("figs/6.4-1.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

def exercise34(linesearch:bool):
    print("Assignment 6.3 and 6.4.")
    from rcracers.simulator.core import simulate
    f = problem.ToyDynamics(False)

    # Build the problem 
    p = problem.ToyProblem()
    
    policy = lambda y,t,log: np.zeros([p.nu,1])
    dynamics = fw_euler(p.cont_dynamics, p.Ts)
    
    # Select initial guess by running an open-loop simulation
    x = simulate(p.x0.reshape([-1,1]), dynamics, p.N, policy=policy).squeeze()
    u = np.zeros([p.N,p.nu])
    _p = np.zeros([p.N,p.ns])
    # x[0,:] = p.x0
    initial_guess = problem.NLIterate(x, u, _p)
    
    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=linesearch, max_iter=100)
    final_iterate = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)
    from given6.homework import animate
    # animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "figs/Assignment6-4"))

    plt.figure()
    plt.plot(final_iterate.solution.x)
    plt.plot(final_iterate.solution.u)
    plt.grid()
    plt.legend(["x_1", "x_2", "x_3", "u"])
    plt.xlabel("Time Step")
    plt.ylabel("State Trajectory")
    plt.title("State Evolution with Time")
    # plt.savefig("figs/6.4-1.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

def exercise56(regularize=False):
    # Build the problem 
    p = problem.ParkingProblem()

    x = np.zeros([p.N+1,p.ns])
    u = np.zeros([p.N,p.nu])
    _p = np.zeros([p.N,p.ns])
    x[0,:] = p.x0
    initial_guess = problem.NLIterate(x, u, _p)

    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=True, max_iter=100, regularize=regularize)
    final_iterate = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)

    from given6.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, f"figs/Assignment6-4-reg{regularize}"))
    animate.animate_positions(logger.iterates, os.path.join(WORKING_DIR, f"figs/parking_regularize-{regularize}"))

    plt.figure()
    plt.plot(final_iterate.solution.x)
    plt.plot(final_iterate.solution.u)
    plt.grid()
    plt.legend(["x_1", "x_2", "x_3", "x_4", "u_1", "u_2"])
    plt.xlabel("Time Step")
    plt.ylabel("State Trajectory")
    plt.title("State Evolution with Time")
    plt.savefig("figs/6.6-1.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    # test_linear_system()
    # exercise2()
    # exercise34(False)
    # exercise34(True)
    # exercise56(regularize=False)
    exercise56(regularize=True)