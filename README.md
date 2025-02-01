# Inverted Pendulum on a Cart (MBSE Project)

This project contains a Python simulation of an inverted pendulum on a cart, 
fulfilling typical MBSE milestone requirements:

1. **Nonlinear Modeling** (ODE-based or PyBullet-based).
2. **Sensor Noise & Filtering** (Gaussian noise, simple low-pass).
3. **Controllers**: PID, Pole-Placement, Nonlinear (Energy Shaping).
4. **Visualization**: Real-time Matplotlib animation with disturbances.
5. **Comparison**: Plot performance for multiple controllers.
6. **Deliverables**: Code, noise model, real-time animation, 
   and adjustable parameters.

## Quick Start

1. Install Python 3.9+ (or Anaconda).  
2. (Conda) Create a new environment:
   ```bash
   conda env create -f environment.yml
   conda activate inverted-pendulum
