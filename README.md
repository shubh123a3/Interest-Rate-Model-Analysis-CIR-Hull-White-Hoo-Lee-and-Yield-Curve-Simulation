# Interest Rate Model Analysis: CIR, Hull-White, Ho-Lee, and Yield Curve Simulation


https://github.com/user-attachments/assets/47a4b2a6-4dfe-4c7a-be5e-af9a52b43cfe


## Overview
This project provides an in-depth analysis of three key interest rate models:
1. **Cox-Ingersoll-Ross (CIR) Model**
2. **Hull-White Model**
3. **Ho-Lee Model**

Additionally, it includes **yield curve generation** using Monte Carlo simulations, fitting interest rate models to the yield curve, and parameter visualization.

## Features
- **Parameter visualization** for CIR and Hull-White models  
- **Monte Carlo simulation** for yield curve generation using Ho-Lee and Hull-White models  
- **Yield curve fitting** to real-world data  
- **Comparative analysis** of different short-rate models  

## Theoretical Background

### 1. Cox-Ingersoll-Ross (CIR) Model
The CIR model is used to describe the evolution of interest rates using the stochastic differential equation:

$$
dR_t = \kappa (\theta - R_t) dt + \sigma \sqrt{R_t} dW_t
$$

where:  
- \( R_t \) is the short-term interest rate  
- \( \kappa \) is the mean reversion speed  
- \( \theta \) is the long-term mean interest rate  
- \( \sigma \) is the volatility of interest rates  
- \( dW_t \) is a Wiener process  

### 2. Hull-White Model
The Hull-White model extends the Vasicek model by allowing time-dependent mean reversion levels:

$$
dR_t = (\theta(t) - \kappa R_t) dt + \sigma dW_t
$$

where \( \theta(t) \) is a deterministic function ensuring the model fits the initial yield curve.

### 3. Ho-Lee Model
The Ho-Lee model assumes a normal distribution of interest rate changes:

$$
dR_t = \theta(t) dt + \sigma dW_t
$$

where \( \theta(t) \) is chosen to fit the yield curve.

## Monte Carlo Simulation for Yield Curve Generation
Monte Carlo methods generate multiple paths of short-rate evolution using the above models. The zero-coupon bond price is computed as:

$$
P(t,T) = \mathbb{E} \left[ e^{-\int_t^T R_s ds} \right]
$$

where \( P(t,T) \) is the price of a zero-coupon bond maturing at \( T \).

## Implementation

### Project Format
The project is structured as follows:
Interest-Rate-Model-Analysis/ │── data/ # Contains sample yield curve data │── src/ # Implementation of models and simulations │ │── cir_model.py # CIR model implementation │ │── hull_white.py # Hull-White model implementation │ │── ho_lee.py # Ho-Lee model implementation │ │── monte_carlo.py # Monte Carlo simulation │── notebooks/ # Jupyter notebooks for visualization │── README.md # Project documentation │── requirements.txt # Dependencies


### Parameter Visualization
- Plotting \( \kappa \), \( \theta \), and \( \sigma \) for CIR and Hull-White models.  
- Simulated short-rate paths for each model.

### Yield Curve Fitting
- Using least squares to optimize \( \theta(t) \) in the Hull-White model.  
- Comparing simulated yield curves with market data.

## Results
- Monte Carlo simulation successfully generates realistic yield curves.  
- Hull-White model provides better calibration to market yield curves.  
- CIR model ensures non-negative interest rates, unlike Ho-Lee.

## Usage
Clone the repository and run:

```bash
git clone https://github.com/shubh123a3/Interest-Rate-Model-Analysis-CIR-Hull-White-Hoo-Lee-and-Yield-Curve-Simulation
cd Interest-Rate-Model-Analysis
python main.py
```
Future Improvements
Implementing Gaussian HJM model for more realistic term structure modeling.
Incorporating machine learning to estimate yield curve parameters.
Extending the Monte Carlo framework to multi-factor models.
References
Cox, J., Ingersoll, J., & Ross, S. (1985). A Theory of the Term Structure of Interest Rates.
Hull, J., & White, A. (1990). Pricing Interest-Rate Derivative Securities.
Ho, T., & Lee, S. (1986). Term Structure Movements and Pricing Interest Rate Contingent Claims.

### ✅ How to Use
1. **Copy the above markdown**  
2. **Paste it into your `README.md` file**  
3. **GitHub will automatically render the LaTeX formulas!**  

Let me know if you need any modifications! 🚀
