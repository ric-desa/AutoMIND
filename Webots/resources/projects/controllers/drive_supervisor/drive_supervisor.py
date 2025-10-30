print("Supervisor controller started")

import traci
import time

# Connect to SUMO inside Webots
traci.init(port=8873)  # Adjust the port if needed

vehicle_id = "11"
depart_time = 0.0

# Add the vehicle at an initial edge
initial_edge = "start_edge_id"  # Replace with your starting edge
traci.vehicle.add(vehicle_id, routeID="", typeID="car", depart=depart_time, departLane="best", departPos="0", departSpeed="0")
traci.vehicle.moveTo(vehicle_id, initial_edge, 0.0)

print("Vehicle added. Enter destination edges to compute routes in real-time.")

while True:
    # Get current edge of the vehicle
    current_edge = traci.vehicle.getRoadID(vehicle_id)
    
    # Ask the user for the destination
    dest_edge = input(f"Current edge: {current_edge}. Enter destination edge (or 'quit' to exit): ")
    if dest_edge.lower() == "quit":
        break

    # Compute the shortest route from current edge to destination
    route_info = traci.simulation.findRoute(current_edge, dest_edge)
    edge_list = [edge[0] for edge in route_info.edges]

    # Update the vehicle's route in real-time
    traci.vehicle.setRoute(vehicle_id, edge_list)
    print(f"Vehicle route updated: {edge_list}\n")

    # Optionally, step the simulation for a few seconds to start movement
    for _ in range(50):
        traci.simulationStep()
        time.sleep(0.1)  # Adjust timing as needed

traci.close()

