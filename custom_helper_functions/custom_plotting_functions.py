import numpy as np
import matplotlib.pyplot as plt

# Plot P(t) and N(t) over t
def time_plot(x, t):
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(t, x[0,:], "b-", label="P(t)")
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('P(t)')
    ax1.tick_params(axis='y', labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("N(t)")
    lns2 = ax2.plot(t, x[1,:], "r-", label="N(t)")
    ax2.tick_params(axis='y', labelcolor="red")
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.set_title("Algae Biomass and Nutritional Density Over Time")
    fig.tight_layout()
    plt.show()

# These two functions multiple (3) plots of P(t) and N(t) over t
def two_scales(ax1, t, x):
    ax2 = ax1.twinx()
    # P
    ax1.plot(t, x[0,:], "b-")
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('P(t)')
    ax1.tick_params(axis='y', labelcolor="blue")
    # N
    ax2.plot(t, x[1,:], "r-")
    ax2.set_ylabel("N(t)")
    ax2.tick_params(axis='y', labelcolor="red")
    return ax1, ax2

def mult_time_plot(t, x1, x2, x3):
    # Create axes
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,3))
    ax1, ax1a = two_scales(ax1, t, x1)
    ax1.set_title("P(0) = {}".format(x1[0,0]))
    ax2, ax2a = two_scales(ax2, t, x2)
    ax2.set_title("P(0) = {}".format(x2[0,0]))
    ax3, ax3a = two_scales(ax3, t, x3)
    ax3.set_title("P(0) = {}".format(x3[0,0]))

    plt.tight_layout()
    plt.show()

#Plotting code for P vs N
def pn_plot(x, plot_eq=False, alpha=0, rho=10**-3, gamma=0, sigma=0.1, betaN0 = 0.1, QN0 = 7.5e-3):
    # Models
    beta = betaN0 / x[1,0]
    Q = QN0 * x[1,0]
    # Plot
    plt.plot(x[0,:], x[1,:])
    plt.plot()
    plt.xlabel("P(t)")
    plt.ylabel("N(t)")
    if plot_eq:
        plt.plot((Q)/(rho*beta*x[1,:]), x[1,:],  color="k")
        plt.axhline((gamma+sigma)/beta, color="k")
    plt.title("Phase-Plane of P(t) and N(t)")
    plt.show()