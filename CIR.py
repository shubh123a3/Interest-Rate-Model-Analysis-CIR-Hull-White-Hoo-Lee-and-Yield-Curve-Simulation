import numpy as np
import matplotlib.pyplot as plt


def GeneratePathsCIREuler(NoOfPaths, NoOfSteps, T, lambd, r0, theta, gamma):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    time = np.zeros([NoOfSteps + 1])
    dt = T / float(NoOfSteps)
    R[:, 0] = r0

    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        R[:, i + 1] = R[:, i] + lambd * (theta - R[:, i]) * dt + gamma * np.sqrt(R[:, i]) * (W[:, i + 1] - W[:, i])
        ### turncanate boundry condtion for non negative intrest rates

        R[:, i + 1] = np.maximum(R[:, i + 1], 0.0)
        time[i + 1] = time[i] + dt

    paths = {"time": time, "R": R}
    return paths


def mainCalculation():
    NoOfPaths = 1
    NoOfSteps = 500
    T = 50.0
    lambd = 0.1
    gamma = 0.05
    r0 = 0.05
    theta = 0.05

    plt.figaspect(1)
    legend = []
    lambdVec = [0.01, 0.2, 5.0]

    for lambdtemp in lambdVec:
        np.random.seed(2)
        Paths = GeneratePathsCIREuler(NoOfPaths, NoOfSteps, T, lambdtemp, r0, theta, gamma)
        legend.append('lambda={0}'.format(lambdtemp))
        timeGrid = Paths["time"]
        R = Paths["R"]
        plt.plot(timeGrid, R.T)
    plt.title("effect of Mean riverson (lambda) ")
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.legend(legend)

    plt.figure(2)
    legend = []
    gammaVec = [0.1, 0.2, 0.3]
    for gammaTemp in gammaVec:
        np.random.seed(2)
        Paths = GeneratePathsCIREuler(NoOfPaths, NoOfSteps, T, lambd, r0, theta, gammaTemp)
        legend.append('gamma={0}'.format(gammaTemp))
        timeGrid = Paths["time"]
        R = Paths["R"]
        plt.plot(timeGrid, np.transpose(R))
    plt.title("effect of Mean gamma(Volality) ")
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.legend(legend)


mainCalculation()