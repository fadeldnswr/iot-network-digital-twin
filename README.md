# Digital Twin-Based Optimization for IoT Communication System
![alt text](image.png)
## Project Description
This project aims to develop a **Digital Twin prototype** of an IoT-based communication system using collected data from sensors (temperature, humidity, RSSI, latency, packet loss). The digital twin is implemented through a **SimPy-based simulation model** that mimics the behavior of a wireless sensor network, including packet transmission delays, environmental impacts, and link quality.

To enhance system performance, the simulation is coupled with **optimization algorithms** such as Genetic Algorithm (GA), Particle Swarm Optimization (PSO), or Tuna Swarm Optimization (TSO), targeting objectives like minimizing communication latency and packet loss or maximizing signal strength (RSSI).

The project is designed for offline data analysis and model validation but serves as the **foundational layer for a full real-time digital twin system**. Future extensions can incorporate live MQTT data streams and feedback mechanisms to form a complete closed-loop cyber-physical system.

## Solved Problems
This project addresses several critical challenges in IoT and communication systems:
1. **Unpredictable Latency**
    Environmental factors like temperature and humidity can impact transmission latency. The simulation quantifies this relationship, enabling proactive configuration.
2. **Packet Loss Modeling**
    By capturing the probabilistic nature of loss due to environmental noise or congestion, the system identifies optimal configurations to minimize data drop rates.
3. **Link Quality Estimation**
    RSSI is used as a proxy for communication health. The model captures its variability and builds empirical functions to predict degradation under changing conditions.
4. **Offline Digital Twin Development**
    Real-world data is used to calibrate a virtual simulation, demonstrating how historical sensor logs can bootstrap a digital twin even when real-time streams are unavailable.
5. **Multi-objective Optimization**
    The integration of metaheuristics (GA/PSO/TSO) allows for efficient search over a complex space of configuration parameters (e.g., transmission intervals, power levels) to achieve a balance between performance metrics.