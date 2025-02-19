import numpy as np
import matplotlib.pyplot as plt


def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta):
    dt = 0.0001
    f0T = lambda t: -(np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)

    r0 = f0T(0.0001)
    theta = lambda t: 1.0 / lambd * (f0T(t + dt) - f0T(t - dt)) / (2.0 * dt) + f0T(t) + eta * eta / (
                2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))

    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    R[:, 0] = r0
    time = np.zeros([NoOfSteps + 1])

    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        R[:, i + 1] = R[:, i] + lambd * (theta(time[i]) - R[:, i]) * dt + eta * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt

    # Outputs
    paths = {"time": time, "R": R}
    return paths


def mainCalculation():
    NoOfPaths = 1
    NoOfSteps = 5000
    T = 50.0
    lambd = 0.5
    eta = 0.01
    # We define a ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-0.05 * T)

    # Effect of mean reversion lambda
    plt.figure(1)
    legend = []
    lambdVec = [-0.01, 0.2, 5.0]
    for lambdTemp in lambdVec:
        np.random.seed(2)
        Paths = GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambdTemp, eta)
        legend.append('lambda={0}'.format(lambdTemp))
        timeGrid = Paths["time"]
        R = Paths["R"]
        plt.plot(timeGrid, np.transpose(R))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.title("efflect of lambda mean revesion")
    plt.legend(legend)

    # Effect of the volatility
    plt.figure(2)
    legend = []
    etaVec = [0.1, 0.2, 0.3]
    for etaTemp in etaVec:
        np.random.seed(2)
        Paths = GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, etaTemp)
        legend.append('eta={0}'.format(etaTemp))
        timeGrid = Paths["time"]
        R = Paths["R"]
        plt.plot(timeGrid, np.transpose(R))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.legend(legend)
    plt.title("efflect Volality in hull white")


mainCalculation()
