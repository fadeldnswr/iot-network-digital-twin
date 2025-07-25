'''
Models for the services module.
This module defines the data models used in the services package.
It includes classes for IoT devices, sensors, gateways, and ESPs,
as well as methods for data processing and analysis.
'''

import simpy
import random as rd
import sys
import os
import pandas as pd
import numpy as np

from datetime import datetime
from src.logging.logging import logging
from src.exception.exception import CustomException
from src.utils.utils import (
  insert_data_to_supabase,
  format_data
)

# Import environment variables
from dotenv import load_dotenv
load_dotenv()

ESP32_REAL_DATA = os.getenv("ESP32_REAL_DATA_PATH")
real_data = pd.read_csv(f"{ESP32_REAL_DATA}esp32_1_data.csv")

# Define real values for temperature and humidity
temp_real = real_data["temperature"].values
humidity_real = real_data["humidity(%)"].values

# Import spline class 
from src.models.math.spline_interpolation import SplineInterpolation

# Import IoT constants
from src.services.constants import *

# Define Sensor class for IoT sensors to send data to gateway
class Sensor:
  '''Sensor class representing an IoT sensor device.'''
  def __init__(self, env: simpy.Environment, temp_series, hum_series, out_store:simpy.Store, name:str, interval:int, minutes:int = 1441):
      self.env = env
      self.interval = interval
      self.out_store = out_store
      self.minutes = minutes
      self.name = name
      
      # Spline from real data
      self.temp_series = temp_series
      self.hum_series = hum_series
      self.process = env.process(self.send_data_to_esp())
  
  def send_data_to_esp(self):
    '''Method to simulate sending data from the sensor.'''
    for i in range(len(self.temp_series)):
      # Define the packet dictionary
      data = {
        "timestamp": self.env.now,
        "sensor": self.name,
        "temperature": round(self.temp_series[i], 2),
        "humidity": round(self.hum_series[i], 2),
      }
      
      # Yield the packet to the store
      yield self.out_store.put(data)
      yield self.env.timeout(self.interval)

# Define ESP32 Class to manage multiple sensors
class ESP32:
  '''ESP32 class representing an IoT ESP32 device.'''
  def __init__(
    self, env:simpy.Environment, in_store:simpy.Store, out_store:simpy.Store,
    packet_loss:float, rssi:float, latency:float, throughput:float, name:str):
      self.env = env
      self.in_store = in_store
      self.out_store = out_store
      self.packet_loss = packet_loss
      self.rssi = rssi
      self.latency = latency
      self.throughput = throughput
      self.name = name
      self.process = env.process(self.send_data_to_gateway())
  
  def send_data_to_gateway(self):
    '''Method to simulate sending data from ESP32 to gateway.'''
    while True:
      # Get the packet from the input store
      packet = yield self.in_store.get()
      
      # Simulate packet metadata
      packet_loss = rd.gauss(self.packet_loss, STD_PACKET_LOSS)
      latency = rd.gauss(self.latency, STD_LATENCY)
      throughput = rd.gauss(self.throughput, STD_THROUGHPUT)
      rssi = rd.gauss(self.rssi, STD_RSSI)
      
      # Forward the packet to the output store as dictionary
      forward_packet = {
        **packet,
        "packet_loss": round(packet_loss, 2),
        "latency": round(latency, 2),
        "throughput": round(throughput, 2),
        "rssi": round(rssi, 2),
        "node": self.name
      }
      yield self.out_store.put(forward_packet)
      logging.info(f"Packet sent from {self.name} to Gateway: {forward_packet}")

# Define Raspberry Pi as Gateway class to receive data from sensors
class Gateway:
  '''Gateway class representing an IoT gateway device.'''
  def __init__(self, env: simpy.Environment, in_store:simpy.Store):
    self.env = env
    self.in_store = in_store
    self.logged_data = [] # Create a list to store logged data
    self.process = env.process(self.receive_data())
  
  def receive_data(self):
    '''Method to simulate receiving data from sensors.'''
    while True:
      logging.info("Waiting for packets from ESPs...")
      packet = yield self.in_store.get()
      yield self.env.timeout(packet["latency"])  # Interval for receiving data based on latency

      # Print the received packet data
      if packet["packet_loss"]:
        print(f"{packet['timestamp']:4} min - {packet['node']}/{packet['sensor']} -> Packet Loss: {packet['packet_loss']}")
      else:
        print(
          f"{packet['timestamp']:4} min - {packet['node']}/{packet['sensor']} -> "
          f"Temperature: {packet['temperature']}°C, Humidity: {packet['humidity']}%, "
          f"Latency: {packet['latency']}ms, Throughput: {packet['throughput']}kbps, "
          f"RSSI: {packet['rssi']}dBm"
        )
      self.logged_data.append(packet) # Store the packet data
      logging.info("Packet has been received and processed successfully.")

# Define IoTSimulation Class to manage the simulation environment
class IoTSimulation:
  '''IoT Simulation class to manage the simulation environment.'''
  def __init__(
    self, sim_time = len(temp_real), interval = SEND_DATA_TIME,
    packet_loss = MEAN_PACKET_LOSS, rssi = MEAN_RSSI,
    latency = MEAN_LATENCY, throughput = MEAN_THROUGHPUT):
      # Create a simulation environment
      self.env = simpy.Environment()
      # Store between sensors and ESPs
      self.sensor_store = [simpy.Store(self.env) for _ in range(NUM_OF_SENSORS)]
      # Store to gateway
      self.store_to_gateway = simpy.Store(self.env)
      
      # Create sensors and ESP nodes
      temp_spline = SplineInterpolation(
        smoothing_factor=100, data=temp_real, rolling_window=None
      )
      _, temp_smooth = temp_spline.generate_spline_function()
      hum_spline = SplineInterpolation(
        smoothing_factor=100, data=humidity_real, rolling_window=None
      )
      _, hum_smooth = hum_spline.generate_spline_function()
      self.sensors = [
        Sensor(
          env=self.env,
          temp_series=temp_smooth,
          hum_series=hum_smooth,
          out_store=self.sensor_store[i],
          name=f"Sensor-{i+1}",
          interval=interval, minutes=SIM_TIME // interval
        ) for i in range(NUM_OF_SENSORS)
      ]
      # Create ESP nodes
      self.esps = [
        ESP32(env=self.env, in_store=self.sensor_store[i], out_store=self.store_to_gateway,
        rssi=rssi, packet_loss=packet_loss, latency=latency, throughput=throughput, name=f"ESP-{i+1}")
        for i in range(NUM_OF_ESP)
      ]
      # Define raspberry pi as a main gateway
      self.main_gateway = Gateway(env=self.env, in_store=self.store_to_gateway)
      # Define the simulation time
      self.sim_time = sim_time
  
  def start_simulation(self):
    '''Method to start the IoT simulation.
    Start the simulation and run it for the defined simulation time.
    '''
    print("Starting IoT Simulation...")
    self.env.run(until=self.sim_time)
    print("Simulation completed.")


# Define a function to run the simulation
if __name__ == "__main__":
  try:
    # Define iot simulation instance
    sim = IoTSimulation()
    
    # Start the simulation
    logging.info("Starting IoT Simulation")
    sim.start_simulation()
    
    # Save the simulation data to CSV
    logging.info("Saving simulation data to Supabase")
    formatted_data = format_data(sim.main_gateway.logged_data, start_time=datetime(2025, 5, 12, 10, 14, 13), seconds=2)
    insert_data_to_supabase(table_name="iot_simulation", data=formatted_data)
    logging.info("IoT Simulation completed")
  except Exception as e:
    raise CustomException(sys, e)