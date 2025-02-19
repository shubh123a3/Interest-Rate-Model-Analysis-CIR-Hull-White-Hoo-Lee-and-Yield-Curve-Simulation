import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# Common Functions
# --------------------------
def f0T(t, P0T):
    dt = 0.01
    return -(np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)


# --------------------------
# Model Path Generators
# --------------------------
def GeneratePathsCIREuler(NoOfPaths, NoOfSteps, T, lambd, r0, theta, gamma):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    dt = T / NoOfSteps
    R[:, 0] = r0

    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.sqrt(dt) * Z[:, i]
        R[:, i + 1] = R[:, i] + lambd * (theta - R[:, i]) * dt + gamma * np.sqrt(R[:, i]) * (W[:, i + 1] - W[:, i])
        R[:, i + 1] = np.maximum(R[:, i + 1], 0.0)

    return {"time": np.linspace(0, T, NoOfSteps + 1), "R": R}


def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta):
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.01, P0T)
    theta = lambda t: 1.0 / lambd * (f0T(t + dt, P0T) - f0T(t - dt, P0T)) / (2.0 * dt) + f0T(t, P0T) + eta * eta / (
                2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))

    # theta = lambda t: 0.1 +t -t
    # print("changed theta")

    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    M = np.zeros([NoOfPaths, NoOfSteps + 1])
    M[:, 0] = 1.0
    R[:, 0] = r0
    time = np.zeros([NoOfSteps + 1])

    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        R[:, i + 1] = R[:, i] + lambd * (theta(time[i]) - R[:, i]) * dt + eta * (W[:, i + 1] - W[:, i])
        M[:, i + 1] = M[:, i] * np.exp((R[:, i + 1] + R[:, i]) * 0.5 * dt)
        time[i + 1] = time[i] + dt

    # Outputs
    paths = {"time": time, "R": R, "M": M}
    return paths


def GeneratePathsHoLeeEuler(NoOfPaths, NoOfSteps, T, P0T, sigma):
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.01, P0T)
    theta = lambda t: (f0T(t + dt, P0T) - f0T(t - dt, P0T)) / (2.0 * dt) + sigma ** 2.0 * t

    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    M = np.zeros([NoOfPaths, NoOfSteps + 1])
    M[:, 0] = 1.0
    R[:, 0] = r0
    time = np.zeros([NoOfSteps + 1])

    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        R[:, i + 1] = R[:, i] + theta(time[i]) * dt + sigma * (W[:, i + 1] - W[:, i])
        M[:, i + 1] = M[:, i] * np.exp((R[:, i + 1] + R[:, i]) * 0.5 * dt)
        time[i + 1] = time[i] + dt

    # Outputs
    paths = {"time": time, "R": R, "M": M}
    return paths

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Interest Rate Models", layout="wide")
st.title("Interest Rate Model Analysis")

# Sidebar Controls
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Analysis Type",
                                ["Parameter Effects", "ZCB Pricing"])

# --------------------------
# Parameter Analysis Section
# --------------------------
if app_mode == "Parameter Effects":
    st.header("Parameter Effect Analysis")

    col1, col2 = st.columns([1, 3])
    with col1:
        model_choice = st.selectbox("Select Model", ["CIR", "Hull-White"])
        param_choice = st.selectbox("Parameter to Analyze",
                                    ["lambda", "gamma"] if model_choice == "CIR" else ["lambda", "eta"])

        param_value = st.number_input(f"Enter {param_choice} value",
                                      min_value=0.01, max_value=10.0, value=0.1, step=0.01)
        if st.button("Add to Analysis"):
            if 'params' not in st.session_state:
                st.session_state.params = []
            st.session_state.params.append((param_choice, param_value))

        if st.button("Clear Parameters"):
            st.session_state.params = []

    with col2:
        if 'params' not in st.session_state:
            st.session_state.params = []

        if len(st.session_state.params) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            for param in st.session_state.params:
                if model_choice == "CIR":
                    paths = GeneratePathsCIREuler(1, 500, 50,
                                                  param[1] if param[0] == "lambda" else 0.1,
                                                  0.05, 0.05,
                                                  param[1] if param[0] == "gamma" else 0.05)
                else:
                    P0T = lambda T: np.exp(-0.05 * T)
                    paths = GeneratePathsHWEuler(1, 1000, 50, P0T,
                                                 param[1] if param[0] == "lambda" else 0.5,
                                                 param[1] if param[0] == "eta" else 0.01)

                ax.plot(paths['time'], paths['R'].T, label=f"{param[0]}={param[1]}")

            ax.set_title(f"{model_choice} Model Parameter Effects")
            ax.set_xlabel("Time")
            ax.set_ylabel("Interest Rate")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            plt.clf()
        else:
            st.info("Add parameters to visualize their effects")

# --------------------------
# ZCB Pricing Section
# --------------------------
else:
    st.header("Zero Coupon Bond Pricing")

    col1, col2 = st.columns([1, 3])
    with col1:
        model_choice = st.selectbox("Select Model", ["Hull-White", "Ho-Lee"])
        a = st.number_input("Exponent Parameter (a)", 0.01, 0.2, 0.05)
        sigma = st.number_input("Volatility (σ)", 0.001, 0.2, 0.007)
        NoOfPaths = st.slider("Number of Paths", 2000, 30000, 1000)
        if model_choice == "Hull-White":
            lambd = st.number_input("Mean Reversion (λ)", 0.001, 1.0, 0.02)

    with col2:
        if st.button("Run Pricing Simulation"):
            P0T = lambda T: np.exp(-a * T)

            if model_choice == "Hull-White":
                paths = GeneratePathsHWEuler(NoOfPaths, 500, 40, P0T, lambd, sigma)
                M = paths["M"]
                ti = paths["time"]
            else:
                paths = GeneratePathsHoLeeEuler(NoOfPaths, 500, 40, P0T,  sigma )
                M = paths["M"]
                ti = paths["time"]
            NoOfSteps=len(ti)-1
            # Calculate ZCB prices
            P_t = np.zeros([NoOfSteps + 1])
            for i in range(0, NoOfSteps + 1):
                P_t[i] = np.mean(1.0 / M[:, i])

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(ti,P0T(ti), label='Market Yield Curve')
            ax.plot(ti,P_t, '--', label='Model Prices')
            ax.set_title(f"ZCB Pricing with {model_choice} Model")
            ax.set_xlabel("Maturity")
            ax.set_ylabel("Price")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            plt.clf()

            # --------------------------
            # Theory Section
            # --------------------------
            with st.expander("Model Theory"):
                if app_mode == "Parameter Effects":

                    st.markdown("""
        **Cox-Ingersoll-Ross (CIR) Model**  
        $$ dr_t = \\lambda(\\theta - r_t)dt + \\gamma\\sqrt{r_t}dW_t $$

        **Hull-White Model**  
        $$ dr_t = (\\theta(t) - \\lambda r_t)dt + \\eta dW_t $$
        """)
                else:

                    st.markdown("""
        **Hull-White Model**  
        $$ P(0,T) = e^{-aT} $$  
        **Ho-Lee Model**  
        $$ dr_t = \\theta(t)dt + \\sigma dW_t $$
        $$ \\theta(t) = \\frac{\\partial f^M(0,t)}{\\partial t} + \\sigma^2 t $$
        """)

            st.sidebar.markdown("---")
            st.sidebar.info("Adjust parameters and click buttons to update visualizations")