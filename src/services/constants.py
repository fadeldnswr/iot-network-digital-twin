'''
Constants for the services module.
This module defines constants used across the services package.
It includes default values for the number of ESPs, sensors, and gateways,
as well as the default port for the MQTT broker.
'''


# Define constant for the default IoT ecosystem configuration
NUM_OF_ESP = 1
NUM_OF_SENSORS = 2
NUM_OF_GATEWAYS = 1

# Define constant for time intervals and simulation parameters
SIM_TIME = 1440 # Simulation time in minutes
SEND_DATA_TIME = 1 # Time interval for sending data in minutes

# Define mean constant for each column data
MEAN_TEMP = 30.909752
MEAN_HUMIDITY = 74.027475
MEAN_LATENCY = 0.277156
MEAN_THROUGHPUT = 0.054000
MEAN_PACKET_LOSS = 0.000000
MEAN_RSSI = -57.561011

# Define std deviation for each column data
STD_TEMP = 8.457271e-01
STD_HUMIDITY = 3.202931e+00
STD_LATENCY = 2.120638e-02
STD_THROUGHPUT = 2.020205e-17
STD_PACKET_LOSS = 0.000000e+00
STD_RSSI = 1.431764e+00