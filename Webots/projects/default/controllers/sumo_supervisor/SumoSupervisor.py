# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SumoSupervisor class inheriting from Supervisor."""

from controller import Supervisor, Node # type: ignore
from Objects import Vehicle, TrafficLight # type: ignore
from WebotsVehicle import WebotsVehicle # type: ignore

import os
import sys
import math
import queue, threading, time, random, uuid

hiddenPosition = 10000


def rotation_from_yaw_pitch_roll(yaw, pitch, roll):
    """Compute the axis-angle rotation from the yaw pitch roll angles"""
    rotation = [0, 0, 1, 0]
    # construct rotation matrix
    # a b c
    # d e f
    # g h i
    a = math.cos(roll) * math.cos(yaw)
    b = -math.sin(roll)
    c = math.cos(roll) * math.sin(yaw)
    d = math.sin(roll) * math.cos(yaw) * math.cos(pitch) + math.sin(yaw) * math.sin(pitch)
    e = math.cos(roll) * math.cos(pitch)
    f = math.sin(roll) * math.sin(yaw) * math.cos(pitch) - math.cos(yaw) * math.sin(pitch)
    g = math.sin(roll) * math.cos(yaw) * math.sin(pitch) - math.sin(yaw) * math.cos(pitch)
    h = math.cos(roll) * math.sin(pitch)
    i = math.sin(roll) * math.sin(yaw) * math.sin(pitch) + math.cos(yaw) * math.cos(pitch)
    # convert it to rotation vector
    cosAngle = 0.5 * (a + e + i - 1.0)
    if math.fabs(cosAngle) > 1:
        return rotation
    else:
        rotation[0] = b - d
        rotation[1] = f - h
        rotation[2] = g - c
        rotation[3] = math.acos(cosAngle)
        # normalize vector
        length = math.sqrt(rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2])
        if length != 0:
            rotation[0] = rotation[0] / length
            rotation[1] = rotation[1] / length
            rotation[2] = rotation[2] / length
        if rotation[0] == 0 and rotation[1] == 0 and rotation[2] == 0:
            return [0, 0, 1, 0]
        else:
            return rotation


class SumoSupervisor (Supervisor):
    """This is the main class that implements the actual interface."""
    
    def __init__(self):
        super().__init__()
        self.prev_lane = None
        self.prev_stop_pos = None

    def get_viewpoint_position_field(self):
        """Look for the 'position' field of the Viewpoint node."""
        children = self.getRoot().getField('children')
        number = children.getCount()
        for i in range(0, number):
            node = children.getMFNode(i)
            if node.getType() == Node.VIEWPOINT:
                return node.getField('position')
        return None

    def get_initial_vehicles(self):
        """Get all the vehicles (both controlled by SUMO and Webots) already present in the world."""
        for i in range(0, self.vehiclesLimit):
            defName = "SUMO_VEHICLE%d" % self.vehicleNumber
            node = self.getFromDef(defName)
            if node:
                self.vehicles[i] = Vehicle(node)
                self.vehicles[i].name.setSFString("SUMO vehicle %i" % self.vehicleNumber)
                self.vehicleNumber += 1
            else:
                break
        for i in range(0, self.vehiclesLimit):
            defName = "WEBOTS_VEHICLE%d" % self.webotsVehicleNumber
            node = self.getFromDef(defName)
            if node:
                self.webotsVehicles[i] = WebotsVehicle(node, self.webotsVehicleNumber)
                self.webotsVehicleNumber += 1
            else:
                break

    def generate_new_vehicle(self, vehicleClass):
        """Generate and import a new vehicle that will be controlled by SUMO."""
        # load the new vehicle
        vehicleString, defName = Vehicle.generate_vehicle_string(self.vehicleNumber, vehicleClass)
        self.rootChildren.importMFNodeFromString(-1, vehicleString)
        nodeRef = self.getFromDef(defName)
        if nodeRef:
            self.vehicles[self.vehicleNumber] = Vehicle(nodeRef)
            self.vehicleNumber += 1

    def get_vehicle_index(self, id, generateIfneeded=True):
        """Look for the vehicle index corresponding to this id (and optionnaly create it if required)."""
        for i in range(0, self.vehicleNumber):
            if self.vehicles[i].currentID == id:
                # the vehicle was already here at last step
                return i
        if not generateIfneeded:
            return -1
        # the vehicle was not present last step
        # check if a corresponding vehicle is already in the simulation
        node = self.getFromDef(id)
        if node and (node.getTypeName() in Vehicle.get_car_models_list() or
                     node.getTypeName() in Vehicle.get_bus_models_list() or
                     node.getTypeName() in Vehicle.get_truck_models_list() or
                     node.getTypeName() in Vehicle.get_motorcycle_models_list()):
            self.vehicles[self.vehicleNumber] = Vehicle(node)
            self.vehicles[self.vehicleNumber].currentID = id
            self.vehicleNumber += 1
            return self.vehicleNumber - 1
        # check if a vehicle is available
        vehicleClass = self.get_vehicle_class(id)
        for i in range(0, self.vehicleNumber):
            if not self.vehicles[i].inUse and self.vehicles[i].vehicleClass == vehicleClass:
                # if a vehicle is available assign it to this id
                self.vehicles[i].currentID = id
                self.vehicles[i].name.setSFString(id)
                return i
        # no vehicle available => generate a new one if limit is not reached
        if self.vehicleNumber < self.vehiclesLimit:
            vehicleClass = self.get_vehicle_class(id)
            self.generate_new_vehicle(vehicleClass)
            return self.vehicleNumber - 1
        return -1

    def get_vehicle_class(self, id):
        """Get the class of the vehicle associated to this id."""
        if id in self.vehiclesClass:
            return self.vehiclesClass[id]
        vehicleClass = Vehicle.get_corresponding_vehicle_class(self.traci.vehicle.getVehicleClass(id))
        self.vehiclesClass[id] = vehicleClass
        return vehicleClass

    def disable_unused_vehicles(self, IdList):
        """Check for all the vehicles currently used if they need to be disabled."""
        for i in range(0, self.vehicleNumber):
            if self.vehicles[i].inUse and self.vehicles[i].currentID not in IdList:
                self.vehicles[i].inUse = False
                self.vehicles[i].name.setSFString("SUMO vehicle %i" % i)
                self.vehicles[i].currentLane = None
                self.vehicles[i].currentRoad = None
                self.vehicles[i].laneChangeStartTime = None
                self.vehicles[i].laneChangeDistance = 0

    def hide_unused_vehicles(self):
        """Hide all the newly unused vehicles."""
        for i in range(0, self.vehicleNumber):
            if not self.vehicles[i].inUse:
                if self.vehicles[i].targetPos[0] != hiddenPosition:
                    self.vehicles[i].targetPos = [hiddenPosition, i * 10, 0.5]
                    self.vehicles[i].currentPos = [hiddenPosition, i * 10, 0.5]
                    self.vehicles[i].currentRot = [0, 0, 1, 0]
                    self.vehicles[i].targetRot = [0, 0, 1, 0]
                    self.vehicles[i].currentAngles = [0, 0, 0]
                    self.vehicles[i].targetAngles = [0, 0, 0]
                    self.vehicles[i].translation.setSFVec3f([hiddenPosition, i * 10, 0.5])
                    self.vehicles[i].node.setVelocity([0, 0, 0, 0, 0, 0])
                    for wheelAngularVelocity in self.vehicles[i].wheelsAngularVelocity:
                        wheelAngularVelocity.setSFVec3f([0, 0, 0])

    def stop_all_vehicles(self):
        """Stop all the vehicles (to be called when controller exits)."""
        for i in range(0, self.vehicleNumber):
            self.vehicles[i].node.setVelocity([0, 0, 0, 0, 0, 0])
            for wheelAngularVelocity in self.vehicles[i].wheelsAngularVelocity:
                wheelAngularVelocity.setSFVec3f([0, 0, 0])

    def get_vehicles_position(self, id, subscriptionResult, step, xOffset, yOffset,
                              maximumLateralSpeed, maximumAngularSpeed, laneChangeDelay):
        """Compute the new desired position and orientation for all the vehicles controlled by SUMO."""
        if not subscriptionResult:
            return
        height = 0.4
        roll = 0.0
        pitch = 0.0
        sumoPos = subscriptionResult[self.traci.constants.VAR_POSITION]
        sumoAngle = subscriptionResult[self.traci.constants.VAR_ANGLE]
        pos = [sumoPos[0] + xOffset, sumoPos[1] + yOffset, height]
        angle = -math.pi * sumoAngle / 180
        dx = math.cos(angle)
        dz = -math.sin(angle)
        yaw = -math.atan2(dx, dz)
        # correct position (origin of the car is not the same in Webots / sumo)
        vehicleLength = subscriptionResult[self.traci.constants.VAR_LENGTH]
        pos[0] += 0.5 * vehicleLength * math.sin(angle)
        pos[1] -= 0.5 * vehicleLength * math.cos(angle)
        # if needed check the vehicle is in the visibility radius
        if self.radius > 0:
            viewpointPosition = self.viewpointPosition.getSFVec3f()
            xDiff = viewpointPosition[0] - pos[0]
            yDiff = viewpointPosition[1] - pos[1]
            zDiff = viewpointPosition[2]
            distance = math.sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff)
            if distance > self.radius:
                index = self.get_vehicle_index(id, generateIfneeded=False)
                if index >= 0:
                    self.vehicles[index].inUse = False
                    self.vehicles[index].currentID = ""
                    self.vehicles[index].name.setSFString("SUMO vehicle %i" % index)
                return
        index = self.get_vehicle_index(id)
        if index >= 0:
            vehicle = self.vehicles[index]
            height = vehicle.wheelRadius
            if self.enableHeight:
                roadID = subscriptionResult[self.traci.constants.VAR_ROAD_ID]
                roadPos = subscriptionResult[self.traci.constants.VAR_LANEPOSITION]
                if roadID.startswith(':'):
                    # this is a lane change it does not contains edge information
                    # in that case, use previous height, roll and pitch
                    height = vehicle.currentPos[2]
                    roll = vehicle.roll
                    pitch = vehicle.pitch
                else:
                    tags = roadID.split('_')
                    del tags[0]  # remove the first one which is the 'id' of the road
                    for tag in tags:
                        if tag.startswith('height'):
                            height = height + float(tag.split('height', 1)[1])
                        elif tag.startswith('roll'):
                            roll = float(tag.split('roll', 1)[1])
                        elif tag.startswith('pitch'):
                            pitch = float(tag.split('pitch', 1)[1])
                    vehicle.pitch = pitch
                    vehicle.roll = roll
                    # ajust height according to the pitch
                    if pitch != 0:
                        height += (roadPos - 0.5 * vehicleLength) * math.sin(pitch)
                    # ajust height according to the roll and lateral position of the vehicle
                    if roll != 0.0:
                        laneIndex = subscriptionResult[self.traci.constants.VAR_LANE_INDEX]
                        laneID = subscriptionResult[self.traci.constants.VAR_LANE_ID]
                        laneWidth = self.traci.lane.getWidth(laneID)
                        edge = self.net.getEdge(roadID)
                        numberOfLane = edge.getLaneNumber()
                        # compute lateral distance from the center of the lane
                        distance = math.fabs((laneIndex - numberOfLane / 2) + 0.5) * laneWidth
                        if laneIndex >= (numberOfLane / 2):
                            height = height - distance * math.sin(roll)
                        else:
                            height = height + distance * math.sin(roll)
            pos[2] = height
            if vehicle.inUse:
                # TODO: once the lane change model of SUMO has been improved
                #       (sub-lane model currently in development phase) we will be able to remove this corrections

                # compute longitudinal (x) and lateral (y) displacement
                diffX = pos[0] - vehicle.targetPos[0]
                diffY = pos[1] - vehicle.targetPos[1]
                x1 = math.cos(-angle) * diffX - math.sin(-angle) * diffY
                y1 = math.sin(-angle) * diffX + math.cos(-angle) * diffY
                # check for lane change
                if (vehicle.currentRoad is not None and
                        vehicle.currentRoad == subscriptionResult[self.traci.constants.VAR_ROAD_ID] and
                        vehicle.currentLane is not None and
                        vehicle.currentLane != subscriptionResult[self.traci.constants.VAR_LANE_INDEX]):
                    vehicle.laneChangeStartTime = self.getTime()
                    vehicle.laneChangeDistance = x1
                x2 = x1
                # artificially add an angle depending on the lateral speed
                artificialAngle = 0
                if y1 > 0.0001:  # don't add the angle if speed is very small as atan2(0.0, 0.0) is unstable
                    # the '0.15' factor was found empirically and should not depend on the simulation
                    artificialAngle = 0.15 * math.atan2(x1, y1)
                if (vehicle.laneChangeStartTime is not None and
                        vehicle.laneChangeStartTime > self.getTime() - laneChangeDelay):  # lane change case
                    ratio = (self.getTime() - vehicle.laneChangeStartTime) / laneChangeDelay
                    ratio = (0.5 + 0.5 * math.sin((ratio - 0.5) * math.pi))
                    p = vehicle.laneChangeDistance * ratio
                    x2 = x1 - (vehicle.laneChangeDistance - p)
                    artificialAngle = math.atan2(-x2, y1)
                # limit lateral speed
                threshold = 0.001 * step * maximumLateralSpeed
                x2 = min(max(x2, -threshold), threshold)
                x3 = math.cos(angle) * x2 - math.sin(angle) * y1
                y3 = math.sin(angle) * x2 + math.cos(angle) * y1
                pos = [x3 + vehicle.targetPos[0], y3 + vehicle.targetPos[1], pos[2]]
                diffYaw = yaw - vehicle.targetAngles[2] - artificialAngle
                # limit angular speed
                diffYaw = (diffYaw + 2 * math.pi) % (2 * math.pi)
                if (diffYaw > math.pi):
                    diffYaw -= 2 * math.pi
                threshold = 0.001 * step * maximumAngularSpeed
                diffYaw = min(max(diffYaw, -threshold), threshold)
                yaw = diffYaw + vehicle.targetAngles[2]
                # tilt motorcycle depending on the angluar speed
                if vehicle.type in Vehicle.get_motorcycle_models_list():
                    threshold = 0.001 * step * maximumLateralSpeed
                    roll -= min(max(diffYaw / (0.001 * step), -0.2), 0.2)
            rot = rotation_from_yaw_pitch_roll(yaw, pitch, roll)
            if not vehicle.inUse:
                # this vehicle was previously not used, move it directly to the correct initial location
                vehicle.inUse = True
                vehicle.currentPos = pos
                vehicle.currentRot = rot
                vehicle.currentAngles = [roll, pitch, yaw]
            else:
                vehicle.currentPos = vehicle.targetPos
                vehicle.currentRot = vehicle.targetRot
                vehicle.currentAngles = vehicle.targetAngles
            # update target and wheels speed
            vehicle.targetPos = pos
            vehicle.targetRot = rot
            vehicle.targetAngles = [roll, pitch, yaw]
            if self.traci.constants.VAR_SPEED in subscriptionResult:
                vehicle.speed = subscriptionResult[self.traci.constants.VAR_SPEED]
            vehicle.currentRoad = subscriptionResult[self.traci.constants.VAR_ROAD_ID]
            vehicle.currentLane = subscriptionResult[self.traci.constants.VAR_LANE_INDEX]

    def update_vehicles_position_and_velocity(self, step, rotateWheels):
        """Update the actual position (using angular and linear velocities) of all the vehicles in Webots."""
        for i in range(0, self.vehicleNumber):
            if self.vehicles[i].inUse:
                self.vehicles[i].translation.setSFVec3f(self.vehicles[i].currentPos)
                self.vehicles[i].rotation.setSFRotation(self.vehicles[i].currentRot)
                velocity = []
                velocity.append(self.vehicles[i].targetPos[0] - self.vehicles[i].currentPos[0])
                velocity.append(self.vehicles[i].targetPos[1] - self.vehicles[i].currentPos[1])
                velocity.append(self.vehicles[i].targetPos[2] - self.vehicles[i].currentPos[2])
                for j in range(0, 3):
                    diffAngle = self.vehicles[i].currentAngles[j] - self.vehicles[i].targetAngles[j]
                    diffAngle = (diffAngle + 2 * math.pi) % (2 * math.pi)
                    if (diffAngle > math.pi):
                        diffAngle -= 2 * math.pi
                    velocity.append(diffAngle)
                velocity[:] = [1000 * x / step for x in velocity]
                self.vehicles[i].node.setVelocity(velocity)
                if rotateWheels:
                    angularVelocity = [0, self.vehicles[i].speed / self.vehicles[i].wheelRadius, 0]
                    for wheelAngularVelocity in self.vehicles[i].wheelsAngularVelocity:
                        wheelAngularVelocity.setSFVec3f(angularVelocity)

    def update_webots_vehicles(self, xOffset, yOffset):
        """Update the position of all the vehicles controlled by Webots in SUMO."""
        for i in range(0, self.webotsVehicleNumber):
            if self.webotsVehicles[i].is_on_road(xOffset, yOffset, self.maxWebotsVehicleDistanceToLane, self.net):
                self.webotsVehicles[i].update_position(self.getTime(), self.net, self.traci, self.sumolib, xOffset, yOffset)
            else:
                # the controlled vehicle is not on any road
                # => we remove it from sumo network
                if self.webotsVehicles[i].name in self.traci.vehicle.getIDList():
                    self.traci.vehicle.remove(self.webotsVehicles[i].name)

    def get_traffic_light(self, IDlist):
        """Get the state of all the traffic lights controlled by SUMO."""
        self.trafficLightNumber = len(IDlist)
        self.trafficLights = {}
        LEDNames = []
        for i in range(0, self.getNumberOfDevices()):
            device = self.getDeviceByIndex(i)
            if device.getNodeType() == Node.LED:
                LEDNames.append(device.getName())
        for i in range(0, self.trafficLightNumber):
            id = IDlist[i]
            self.trafficLights[id] = TrafficLight()
            self.trafficLights[id].lightNumber = len(self.traci.trafficlight.getRedYellowGreenState(id))
            for j in range(0, self.trafficLights[id].lightNumber):
                trafficLightNode = self.getFromDef("TLS_" + id + "_" + str(j))
                if trafficLightNode is not None:
                    self.trafficLights[id].trafficLightRecognitionColors[j] = trafficLightNode.getField('recognitionColors')
                ledName = id + "_" + str(j) + "_"
                if ledName + 'r' in LEDNames:
                    self.trafficLights[id].LED[3 * j + 0] = self.getDevice(ledName + 'r')
                else:
                    self.trafficLights[id].LED[3 * j + 0] = None
                if ledName + 'y' in LEDNames:
                    self.trafficLights[id].LED[3 * j + 1] = self.getDevice(ledName + 'y')
                else:
                    self.trafficLights[id].LED[3 * j + 1] = None
                if ledName + 'g' in LEDNames:
                    self.trafficLights[id].LED[3 * j + 2] = self.getDevice(ledName + 'g')
                else:
                    self.trafficLights[id].LED[3 * j + 2] = None

    def update_traffic_light_state(self, id, states):
        """Update the traffic lights state in Webots."""
        # update light LED state if traffic light state has changed
        currentState = states[self.traci.constants.TL_RED_YELLOW_GREEN_STATE]
        if self.trafficLights[id].previousState != currentState:
            self.trafficLights[id].previousState = currentState
            for j in range(0, self.trafficLights[id].lightNumber):
                # Update red LED if it exists
                if self.trafficLights[id].LED[3 * j + 0]:
                    if currentState[j] == 'r' or currentState[j] == 'R':
                        self.trafficLights[id].LED[3 * j + 0].set(1)
                        # update recognition colors
                        if j in self.trafficLights[id].trafficLightRecognitionColors:
                            self.trafficLights[id].trafficLightRecognitionColors[j].setMFColor(1, [1, 0, 0])
                    else:
                        self.trafficLights[id].LED[3 * j + 0].set(0)
                # Update yellow LED if it exists
                if self.trafficLights[id].LED[3 * j + 1]:
                    if currentState[j] == 'y' or currentState[j] == 'Y':
                        self.trafficLights[id].LED[3 * j + 1].set(1)
                        # update recognition colors
                        if j in self.trafficLights[id].trafficLightRecognitionColors:
                            self.trafficLights[id].trafficLightRecognitionColors[j].setMFColor(1, [1, 0.5, 0])
                    else:
                        self.trafficLights[id].LED[3 * j + 1].set(0)
                # Update green LED if it exists
                if self.trafficLights[id].LED[3 * j + 2]:
                    if currentState[j] == 'g' or currentState[j] == 'G':
                        self.trafficLights[id].LED[3 * j + 2].set(1)
                        # update recognition colors
                        if j in self.trafficLights[id].trafficLightRecognitionColors:
                            self.trafficLights[id].trafficLightRecognitionColors[j].setMFColor(1, [0, 1, 0])
                    else:
                        self.trafficLights[id].LED[3 * j + 2].set(0)

    def run(self, port, disableTrafficLight, directory, step, rotateWheels,
            maxVehicles, radius, enableHeight, useDisplay, displayRefreshRate,
            displayZoom, displayFitSize, maximumLateralSpeed, maximumAngularSpeed,
            laneChangeDelay, traci, sumolib):
        """Main loop function."""
        try:
            print('Connect to SUMO... This operation may take a few seconds.')
            self.step(step)
            traci.init(port, numRetries=20)
        except Exception:
            sys.exit('Unable to connect to SUMO, please make sure any previous instance of SUMO is closed.\n You can try'
                     ' changing SUMO port using the "--port" argument.')


        print(">>> SUMO connected successfully! Now running custom logic...")

        base_comm_dir = "C:\\Users\\ACER\\Documents\\CS\\Multimodal Interaction"
        speed_file = os.path.join(base_comm_dir, "speed.txt")
        accel_file = os.path.join(base_comm_dir, "acceleration.txt")

        # ensure files exist and are empty initially
        for p in (speed_file, accel_file):
            try:
                with open(p, "w") as f:
                    f.write("") 
            except Exception as e:
                print(f"Could not create {p}: {e}")

        self.traci = traci
        self.sumolib = sumolib
        self.radius = radius
        self.enableHeight = enableHeight
        self.sumoClosed = False
        self.temporaryDirectory = directory
        self.rootChildren = self.getRoot().getField('children')
        self.viewpointPosition = self.get_viewpoint_position_field()
        self.maxWebotsVehicleDistanceToLane = 15
        self.webotsVehicleNumber = 0
        self.webotsVehicles = {}
        self.vehicleNumber = 0
        self.vehicles = {}
        self.vehiclesLimit = maxVehicles
        self.vehiclesClass = {}

        directory = os.path.normpath(directory)

        # for backward compatibility
        if self.traci.constants.TRACI_VERSION <= 15:
            self.traci.trafficlight = self.traci.trafficlights

        # get sumo vehicles already present in the world
        self.get_initial_vehicles()

        # parse the net and get the offsets
        self.net = sumolib.net.readNet(os.path.join(directory, 'sumo.net.xml'))
        xOffset = -self.net.getLocationOffset()[0]
        yOffset = -self.net.getLocationOffset()[1]

        # Load plugin to the generic SUMO Supervisor (if any)
        self.usePlugin = False
        if os.path.exists(os.path.join(directory, 'plugin.py')):
            self.usePlugin = True
            sys.path.append(directory)
            import plugin # type: ignore
            sumoSupervisorPlugin = plugin.SumoSupervisorPlugin(self, self.traci, self.net)

        # Get all the LEDs of the traffic lights
        if not disableTrafficLight:
            trafficLightsList = self.traci.trafficlight.getIDList()
            self.get_traffic_light(trafficLightsList)
            for id in trafficLightsList:
                # subscribe to traffic lights state
                self.traci.trafficlight.subscribe(id, [self.traci.constants.TL_RED_YELLOW_GREEN_STATE])

        # Subscribe to new vehicles entering the simulation
        self.traci.simulation.subscribe([
            self.traci.constants.VAR_DEPARTED_VEHICLES_IDS,
            self.traci.constants.VAR_MIN_EXPECTED_VEHICLES,
            self.traci.constants.VAR_ARRIVED_VEHICLES_IDS,
        ])

        # Create the vehicle variable subscription list
        self.vehicleVariableList = [
            self.traci.constants.VAR_POSITION,
            self.traci.constants.VAR_ANGLE,
            self.traci.constants.VAR_LENGTH,
            self.traci.constants.VAR_ROAD_ID,
            self.traci.constants.VAR_LANE_INDEX
        ]
        if rotateWheels:
            self.vehicleVariableList.append(self.traci.constants.VAR_SPEED)
        if enableHeight:
            self.vehicleVariableList.extend([
                self.traci.constants.VAR_ROAD_ID,
                self.traci.constants.VAR_LANEPOSITION,
                self.traci.constants.VAR_LANE_ID
            ])

        dest_queue = queue.Queue()

        def watch_dest_file(file_path, q):
            """Watches a plain-text file and puts new lines into the queue.
               Once lines are enqueued the file is truncated so entries are processed only once."""
            last_size = 0
            while True:
                try:
                    if not os.path.exists(file_path):
                        # ensure file exists
                        with open(file_path, "w") as _:
                            pass
                    with open(file_path, "r+") as f:
                        content = f.read().strip()
                        if content:
                            # enqueue each non-empty line
                            for line in content.splitlines():
                                line = line.strip()
                                if line:
                                    print(f"[watcher] enqueuing: {line}")
                                    q.put(line)
                            # truncate file after reading to avoid duplicates
                            f.seek(0)
                            f.truncate(0)
                except Exception as e:
                    # keep watcher alive on errors
                    print(f"watch_dest_file error: {e}")
                time.sleep(0.5)  # check every 0.5s

        # start watcher thread
        # threading.Thread(target=watch_dest_file, args=("new_dest.txt", dest_queue), daemon=True).start()

        # try:
        #     vehicle_id = "0"
        #     depart_time = 0.0
        #     prev_lane, prev_pos = "", 0.0

        #     start_edge = "-19"
        #     dest_edge  = "19"
        #     # Compute a route automatically from start to destination
        #     route = traci.simulation.findRoute(start_edge, dest_edge)
        #     # print(f"Route: {route}")
        #     edge_list = route.edges
        #     # edge_list = [e[0] for e in route.edges]
        #     # print(f"edge_list: {edge_list}")
        #     # start_length = traci.lane.getLength(start_edge + "_0")  # first lane of edge
        #     # print(f"edge_length: {start_length}")

        #     # Add the vehicle to the network with that route
        #     traci.vehicle.add(vehicle_id, routeID="", # typeID="carType",
        #                     depart=depart_time, departLane="best",
        #                     departPos="100.0", departSpeed="0")
        #     self.traci.vehicle.subscribe(vehicle_id, self.vehicleVariableList)
        #     traci.vehicle.setRoute(vehicle_id, edge_list)
        #     # traci.vehicle.moveTo("0", "-26_0", 250.0)
        #     dest_length = traci.lane.getLength(dest_edge + "_0")
        #     traci.vehicle.setStop(vehicle_id, dest_edge, dest_length-0.1) # duration=1)  # duration in sec


        #     print(f"Added '{vehicle_id}'")
        #     # print(f"Added '{vehicle_id}' with route: {edge_list}")

        # except Exception as e:
        #     print(f"Error creating vehicle: {e}")
                
            # NOTE: process new destinations
            # while not dest_queue.empty():
            #     stop_flag = traci.vehicle.getStopState(vehicle_id)
            #     print(f"vehicle stopped: {stop_flag}")
            #     new_dest = dest_queue.get().strip()
            #     if new_dest.lower() == "quit":
            #         continue
            #     current_edge = traci.vehicle.getRoadID(vehicle_id)
            #     try:
            #         lane, stop_pos = new_dest.split()
            #         print("lane initial", lane, stop_pos)
            #         if lane == prev_lane and float(stop_pos) < prev_pos:
            #             stop_flag = False
            #             inter_lane = -int(lane)
            #             inter_stop_pos = "5.0"
            #             print("inter:",inter_lane, inter_stop_pos)
            #             route = traci.simulation.findRoute(current_edge, inter_lane)
            #             traci.vehicle.resume(vehicle_id)
            #             traci.vehicle.setRoute(vehicle_id, list(route.edges))
            #             traci.vehicle.setStop(vehicle_id, inter_lane, float(inter_stop_pos))
            #             current_edge = traci.vehicle.getRoadID(vehicle_id)
            #             print(f"current edge: {current_edge}")
            #             dest_queue.put(new_dest)
            #             prev_lane = inter_lane
            #             prev_pos = float(inter_stop_pos)
                        
            #         if stop_flag:
            #             route = traci.simulation.findRoute(current_edge, lane)
            #             traci.vehicle.resume(vehicle_id)
            #             traci.vehicle.setRoute(vehicle_id, list(route.edges))
            #             prev_lane = lane
            #             prev_pos = float(stop_pos)
            #             if not stop_pos:
            #                 edge_length = traci.lane.getLength(new_dest + "_0")
            #                 traci.vehicle.setStop(vehicle_id, new_dest, edge_length-1)
            #             else:
            #                 traci.vehicle.setStop(vehicle_id, lane, float(stop_pos))
            #             print(f"Vehicle {vehicle_id} route updated to {lane}")
            #     except Exception as e:
            #         print(f"Error updating route: {e}")

            

            # def monitor_vehicle_stop(vehicle_id, dest_queue):
            #     i=0
            #     intermediate_done = True
            #     while True:
            #         i+=1
            #         if not dest_queue.empty():
            #             stop_flag = traci.vehicle.getStopState(vehicle_id)
            #             if stop_flag and intermediate_done:  # vehicle is stopped
            #                 new_dest = dest_queue.get().strip()
            #                 if new_dest.lower() == "quit":
            #                     continue

            #                 current_edge = traci.vehicle.getRoadID(vehicle_id)
            #                 try:
            #                     lane, stop_pos = new_dest.split()
            #                     lane = str(lane)
            #                     stop_pos = float(stop_pos)
            #                     print("lane target", lane, stop_pos)

            #                     # Handle intermediate stop if going backwards along same lane
            #                     if lane == self.prev_lane and stop_pos < self.prev_stop_pos:
            #                         # create a small intermediate stop to avoid SUMO removing the vehicle
            #                         inter_lane = str(-int(lane))  
            #                         inter_stop_pos = 5.0
            #                         print("inter:",inter_lane, inter_stop_pos)
            #                         print(f'i at inter:{i}')
            #                         print("||")
            #                         route = traci.simulation.findRoute(current_edge, inter_lane)
            #                         traci.vehicle.resume(vehicle_id)
            #                         traci.vehicle.setRoute(vehicle_id, list(route.edges))
            #                         traci.vehicle.setStop(vehicle_id, inter_lane, inter_stop_pos)
            #                         intermediate_done = False
            #                         # put the original destination back to queue
            #                         dest_queue.put(new_dest)
            #                         self.prev_lane = inter_lane
            #                         self.prev_stop_pos = inter_stop_pos
            #                         continue  # wait for intermediate stop

            #                     # Normal stop at destination
            #                     print(f'i at final:{i}')
            #                     print(f"target lane: {lane}, prev_lane: {self.prev_lane, self.prev_stop_pos}")
            #                     route = traci.simulation.findRoute(current_edge, lane)
            #                     traci.vehicle.resume(vehicle_id)
            #                     traci.vehicle.setRoute(vehicle_id, list(route.edges))
            #                     traci.vehicle.setStop(vehicle_id, lane, stop_pos)
            #                     self.prev_lane = lane
            #                     self.prev_stop_pos = stop_pos
            #                     intermediate_done = True
            #                     print(f"Vehicle {vehicle_id} route updated to {lane} stop at {stop_pos}")
            #                     print("|")

            #                 except Exception as e:
            #                     print(f"Error updating route: {e}")

            #             if not intermediate_done:
            #                 stop_flag = traci.vehicle.getStopState(vehicle_id)
            #                 if stop_flag:
            #                     intermediate_done = True

            #         time.sleep(0.05)  # small delay to avoid hogging CPU

            # threading.Thread(target=monitor_vehicle_stop, args=(vehicle_id, dest_queue), daemon=True).start()

        # Destination processing thread - consumes dest_queue sequentially and applies them safely

        def process_destinations(vehicle_id, q):
            """
            Dedicated thread to process destination requests one-by-one.
            Each queue element is either:
                - "<edge_id> <pos>"  (pos optional)
                - "<edge_id>"        (no position; means end of lane)
                - "quit"             (ignored)
            Behavior:
                - If dest is on same edge but behind current lane position, create an intermediate
                detour (searching the net for a reachable alternate edge), set a small stop there,
                wait until the vehicle actually stops, then resume and set the final destination.
                - Otherwise, set the route immediately and schedule a stop at the target edge/position.
            """
            while True:
                new_dest_raw = q.get()  # blocking, ensures sequential processing
                if not new_dest_raw:
                    continue
                new_dest = new_dest_raw.strip()
                if not new_dest:
                    continue
                if new_dest.lower() == "quit":
                    continue

                try:
                    parts = new_dest.split()
                    target_edge = parts[0]
                    stop_pos = None
                    if len(parts) > 1:
                        try:
                            stop_pos = float(parts[1])
                        except Exception:
                            stop_pos = None

                    # Ensure vehicle exists
                    try:
                        vehicles = self.traci.vehicle.getIDList()
                    except Exception as e:
                        print(f"[dest_manager] error getting vehicle list: {e}")
                        time.sleep(0.2)
                        q.put(new_dest_raw)  # retry later
                        continue

                    if vehicle_id not in vehicles:
                        # vehicle might be not yet added or removed; retry later
                        print(f"[dest_manager] vehicle {vehicle_id} not present; retrying later")
                        time.sleep(0.5)
                        q.put(new_dest_raw)
                        continue

                    # get current edge and lane position
                    try:
                        current_edge = self.traci.vehicle.getRoadID(vehicle_id)
                    except Exception as e:
                        print(f"[dest_manager] cannot read current edge: {e}")
                        time.sleep(0.2)
                        q.put(new_dest_raw)
                        continue

                    # lane position in meters from start of lane (if available)
                    try:
                        current_lane_pos = self.traci.vehicle.getLanePosition(vehicle_id)
                    except Exception:
                        current_lane_pos = None

                    # If destination is on same edge and behind current position -> need intermediate detour
                    same_edge_and_behind = False
                    if current_edge == target_edge and (stop_pos is not None and current_lane_pos is not None):
                        try:
                            # if requested stop_pos is less than current pos -> it's behind
                            same_edge_and_behind = float(stop_pos) < float(current_lane_pos)
                        except Exception:
                            same_edge_and_behind = False

                    # Always attempt to set a new route (resume if stopped)
                    try:
                        self.traci.vehicle.resume(vehicle_id)
                    except Exception:
                        pass

                    if same_edge_and_behind:
                        # find a reachable intermediate edge that is not the same edge
                        inter_edge = None
                        inter_stop_pos = 2.0  # small forward pos on intermediate edge
                        try:
                            # try numeric negation first (e.g. "17" -> "-17", "-3" -> "3")
                            try:
                                inv = str(-int(target_edge))
                            except Exception:
                                # fallback for non-pure-numeric ids like "edge_17"
                                if target_edge.startswith('-'):
                                    inv = target_edge[1:]
                                else:
                                    inv = '-' + target_edge

                            # verify there's a route to the inverse edge
                            try:
                                route = self.traci.simulation.findRoute(current_edge, inv)
                                if route and len(route.edges) > 0:
                                    inter_edge = inv
                            except Exception:
                                inter_edge = None
                        except Exception:
                            inter_edge = None

                        if inter_edge is None:
                            # fallback: if we cannot find any intermediate edge, just attempt to route directly (best-effort)
                            print(f"[dest_manager] no intermediate edge found for {new_dest}; trying direct route")
                            try:
                                route = self.traci.simulation.findRoute(current_edge, target_edge)
                                if route and len(route.edges) > 0:
                                    self.traci.vehicle.setRoute(vehicle_id, list(route.edges))
                                    if stop_pos is None:
                                        # use end-of-lane if no stop pos given
                                        try:
                                            lane_len = self.traci.lane.getLength(f"{target_edge}_0")
                                            self.traci.vehicle.setStop(vehicle_id, target_edge, lane_len - 0.1)
                                        except Exception:
                                            # final fallback: setStop without pos
                                            self.traci.vehicle.setStop(vehicle_id, target_edge, 0.0)
                                    else:
                                        self.traci.vehicle.setStop(vehicle_id, target_edge, stop_pos)
                                    print(f"[dest_manager] applied direct route to {target_edge} (no inter)")
                                    continue
                            except Exception as e:
                                print(f"[dest_manager] failed to route directly: {e}; will retry later")
                                time.sleep(0.5)
                                q.put(new_dest_raw)
                                continue

                        # Set intermediate route + stop, wait until vehicle stops there, then continue to final dest
                        try:
                            route = self.traci.simulation.findRoute(current_edge, inter_edge)
                            if route and len(route.edges) > 0:
                                self.traci.vehicle.setRoute(vehicle_id, list(route.edges))
                                # ensure vehicle is moving towards intermediate
                                try:
                                    self.traci.vehicle.resume(vehicle_id)
                                except Exception:
                                    pass
                                # schedule the intermediate stop
                                try:
                                    self.traci.vehicle.setStop(vehicle_id, inter_edge, inter_stop_pos)
                                    print(f"[dest_manager] set intermediate stop on {inter_edge} at {inter_stop_pos} before final dest {target_edge}")
                                except Exception as e:
                                    print(f"[dest_manager] failed to set intermediate stop: {e}")
                                    # even if stop fails, continue to try final dest later
                                    q.put(new_dest_raw)
                                    continue

                                # wait until vehicle actually stops at the intermediate stop
                                waited = 0.0
                                max_wait = 60.0  # seconds (safety)
                                while True:
                                    try:
                                        stop_state = self.traci.vehicle.getStopState(vehicle_id)
                                    except Exception:
                                        stop_state = 0
                                    # stop_state == 0 -> not stopped; >0 indicates different stop reasons
                                    if stop_state != 0:
                                        # vehicle stopped at intermediate
                                        print(f"[dest_manager] vehicle {vehicle_id} stopped at intermediate {inter_edge}")
                                        break
                                    time.sleep(0.2)
                                    waited += 0.2
                                    if waited > max_wait:
                                        print("[dest_manager] timeout waiting for intermediate stop; proceeding to final dest")
                                        break

                                # Now apply final destination immediately (don't re-enqueue)
                                try:
                                    current_edge = self.traci.vehicle.getRoadID(vehicle_id)
                                except Exception:
                                    current_edge = route.edges[-1] if route and len(route.edges) > 0 else current_edge

                                try:
                                    final_route = self.traci.simulation.findRoute(current_edge, target_edge)
                                    if final_route and len(final_route.edges) > 0:
                                        self.traci.vehicle.resume(vehicle_id)
                                        self.traci.vehicle.setRoute(vehicle_id, list(final_route.edges))
                                        if stop_pos is None:
                                            try:
                                                lane_len = self.traci.lane.getLength(f"{target_edge}_0")
                                                self.traci.vehicle.setStop(vehicle_id, target_edge, lane_len - 0.1)
                                            except Exception:
                                                self.traci.vehicle.setStop(vehicle_id, target_edge, 0.0)
                                        else:
                                            self.traci.vehicle.setStop(vehicle_id, target_edge, stop_pos)
                                        print(f"[dest_manager] applied final destination {target_edge} stop {stop_pos}")
                                    else:
                                        print(f"[dest_manager] could not compute final route to {target_edge}; will retry")
                                        q.put(new_dest_raw)
                                except Exception as e:
                                    print(f"[dest_manager] error applying final route: {e}; will retry")
                                    q.put(new_dest_raw)
                                continue
                            else:
                                print(f"[dest_manager] no route to intermediate {inter_edge}; will retry later")
                                time.sleep(0.5)
                                q.put(new_dest_raw)
                                continue
                        except Exception as e:
                            print(f"[dest_manager] error during intermediate handling: {e}; will retry")
                            time.sleep(0.5)
                            q.put(new_dest_raw)
                            continue

                    # NORMAL CASE: target is different edge or ahead on same edge
                    try:
                        route = self.traci.simulation.findRoute(current_edge, target_edge)
                        # print(f"calculating route {current_edge} → {target_edge}: {route}")
                        if route and len(route.edges) > 0:
                            self.traci.vehicle.setRoute(vehicle_id, list(route.edges))
                            # ensure vehicle is moving
                            try:
                                self.traci.vehicle.resume(vehicle_id)
                            except Exception:
                                pass
                            if stop_pos is None:
                                # set stop at very end of lane if user didn't specify a position
                                try:
                                    lane_len = self.traci.lane.getLength(f"{target_edge}_0")
                                    self.traci.vehicle.setStop(vehicle_id, target_edge, lane_len - 0.1)
                                except Exception:
                                    self.traci.vehicle.setStop(vehicle_id, target_edge, 0.0)
                            else:
                                self.traci.vehicle.setStop(vehicle_id, target_edge, stop_pos)
                            print(f"[dest_manager] set route to {target_edge} stop {stop_pos}")
                        else:
                            print(f"[dest_manager] could not compute route to {target_edge}; will retry later")
                            time.sleep(0.5)
                            q.put(new_dest_raw)
                    except Exception as e:
                        print(f"[dest_manager] error setting route to {target_edge}: {e}; will retry")
                        time.sleep(0.5)
                        q.put(new_dest_raw)

                except Exception as e:
                    print(f"[dest_manager] unexpected error processing new_dest '{new_dest}': {e}")
                    # don't drop the destination; requeue it to try again later
                    try:
                        q.put(new_dest_raw)
                    except Exception:
                        pass
                    time.sleep(0.5)

        
        def process_destinations2(vehicle_id, q):
            """
            Dedicated thread to process destination requests one-by-one.
            Each queue element:
                - "<edge_id> <pos>"  (pos optional)
                - "<edge_id>"        (no position; means end of lane)
                - "quit"             (ignored)
            """
            while True:
                new_dest_raw = q.get()  # blocking, ensures sequential processing
                try:
                    if not new_dest_raw:
                        q.task_done()
                        continue
                    new_dest = new_dest_raw.strip()
                    if not new_dest:
                        q.task_done()
                        continue
                    if new_dest.lower() == "quit":
                        q.task_done()
                        continue

                    print(f"[dest_manager] processing: '{new_dest}'")

                    parts = new_dest.split()
                    target_edge = parts[0]
                    stop_pos = None
                    if len(parts) > 1:
                        try:
                            stop_pos = float(parts[1])
                        except Exception:
                            stop_pos = None

                    # Ensure vehicle exists
                    try:
                        vehicles = self.traci.vehicle.getIDList()
                    except Exception as e:
                        print(f"[dest_manager] error getting vehicle list: {e}")
                        time.sleep(0.2)
                        q.put(new_dest_raw)  # retry later
                        q.task_done()
                        continue

                    if vehicle_id not in vehicles:
                        print(f"[dest_manager] vehicle {vehicle_id} not present; retrying later")
                        time.sleep(0.5)
                        q.put(new_dest_raw)
                        q.task_done()
                        continue

                    # get current edge and lane position
                    try:
                        current_edge = self.traci.vehicle.getRoadID(vehicle_id)
                    except Exception as e:
                        print(f"[dest_manager] cannot read current edge: {e}")
                        time.sleep(0.2)
                        q.put(new_dest_raw)
                        q.task_done()
                        continue

                    try:
                        current_lane_pos = self.traci.vehicle.getLanePosition(vehicle_id)
                    except Exception:
                        current_lane_pos = None

                    same_edge_and_behind = False
                    if current_edge == target_edge and (stop_pos is not None and current_lane_pos is not None):
                        try:
                            same_edge_and_behind = float(stop_pos) < float(current_lane_pos)
                        except Exception:
                            same_edge_and_behind = False

                    # resume if stopped so it can accept route changes
                    try:
                        self.traci.vehicle.resume(vehicle_id)
                    except Exception:
                        pass

                    if same_edge_and_behind:
                        # find intermediate edge
                        inter_edge = None
                        inter_stop_pos = 5.0
                        try:
                            for e in self.net.getEdges():
                                eid = e.getID()
                                if eid == target_edge:
                                    continue
                                try:
                                    route = self.traci.simulation.findRoute(current_edge, eid)
                                    if route and len(route.edges) > 0:
                                        inter_edge = eid
                                        break
                                except Exception:
                                    continue
                        except Exception:
                            inter_edge = None

                        if inter_edge is None:
                            print(f"[dest_manager] no intermediate edge found for {new_dest}; trying direct route")
                            try:
                                route = self.traci.simulation.findRoute(current_edge, target_edge)
                                if route and len(route.edges) > 0:
                                    self.traci.vehicle.setRoute(vehicle_id, list(route.edges))
                                    if stop_pos is None:
                                        try:
                                            lane_len = self.traci.lane.getLength(f"{target_edge}_0")
                                            self.traci.vehicle.setStop(vehicle_id, target_edge, lane_len - 0.1)
                                        except Exception:
                                            self.traci.vehicle.setStop(vehicle_id, target_edge, 0.0)
                                    else:
                                        self.traci.vehicle.setStop(vehicle_id, target_edge, stop_pos)
                                    print(f"[dest_manager] applied direct route to {target_edge} (no inter)")
                                    q.task_done()
                                    continue
                            except Exception as e:
                                print(f"[dest_manager] failed to route directly: {e}; will retry later")
                                time.sleep(0.5)
                                q.put(new_dest_raw)
                                q.task_done()
                                continue

                        # Set intermediate route + stop
                        try:
                            route = self.traci.simulation.findRoute(current_edge, inter_edge)
                            if route and len(route.edges) > 0:
                                self.traci.vehicle.setRoute(vehicle_id, list(route.edges))
                                try:
                                    self.traci.vehicle.resume(vehicle_id)
                                except Exception:
                                    pass
                                try:
                                    self.traci.vehicle.setStop(vehicle_id, inter_edge, inter_stop_pos)
                                    print(f"[dest_manager] set intermediate stop on {inter_edge} at {inter_stop_pos} before final dest {target_edge}")
                                except Exception as e:
                                    print(f"[dest_manager] failed to set intermediate stop: {e}")
                                    q.put(new_dest_raw)
                                    q.task_done()
                                    continue

                                # wait until vehicle stops (or timeout)
                                waited = 0.0
                                max_wait = 60.0
                                while True:
                                    try:
                                        stop_state = self.traci.vehicle.getStopState(vehicle_id)
                                    except Exception:
                                        stop_state = 0
                                    if stop_state != 0:
                                        print(f"[dest_manager] vehicle {vehicle_id} stopped at intermediate {inter_edge}")
                                        break
                                    time.sleep(0.2)
                                    waited += 0.2
                                    if waited > max_wait:
                                        print("[dest_manager] timeout waiting for intermediate stop; proceeding to final dest")
                                        break

                                # apply final destination
                                try:
                                    current_edge = self.traci.vehicle.getRoadID(vehicle_id)
                                except Exception:
                                    current_edge = route.edges[-1] if route and len(route.edges) > 0 else current_edge

                                try:
                                    final_route = self.traci.simulation.findRoute(current_edge, target_edge)
                                    if final_route and len(final_route.edges) > 0:
                                        self.traci.vehicle.resume(vehicle_id)
                                        self.traci.vehicle.setRoute(vehicle_id, list(final_route.edges))
                                        if stop_pos is None:
                                            try:
                                                lane_len = self.traci.lane.getLength(f"{target_edge}_0")
                                                self.traci.vehicle.setStop(vehicle_id, target_edge, lane_len - 0.1)
                                            except Exception:
                                                self.traci.vehicle.setStop(vehicle_id, target_edge, 0.0)
                                        else:
                                            self.traci.vehicle.setStop(vehicle_id, target_edge, stop_pos)
                                        print(f"[dest_manager] applied final destination {target_edge} stop {stop_pos}")
                                    else:
                                        print(f"[dest_manager] could not compute final route to {target_edge}; will retry")
                                        q.put(new_dest_raw)
                                except Exception as e:
                                    print(f"[dest_manager] error applying final route: {e}; will retry")
                                    q.put(new_dest_raw)
                                q.task_done()
                                continue
                            else:
                                print(f"[dest_manager] no route to intermediate {inter_edge}; will retry later")
                                time.sleep(0.5)
                                q.put(new_dest_raw)
                                q.task_done()
                                continue
                        except Exception as e:
                            print(f"[dest_manager] error during intermediate handling: {e}; will retry")
                            time.sleep(0.5)
                            q.put(new_dest_raw)
                            q.task_done()
                            continue

                    # NORMAL CASE: different edge or ahead on same edge
                    try:
                        route = self.traci.simulation.findRoute(current_edge, target_edge)
                        if route and len(route.edges) > 0:
                            self.traci.vehicle.setRoute(vehicle_id, list(route.edges))
                            try:
                                self.traci.vehicle.resume(vehicle_id)
                            except Exception:
                                pass
                            if stop_pos is None:
                                try:
                                    lane_len = self.traci.lane.getLength(f"{target_edge}_0")
                                    self.traci.vehicle.setStop(vehicle_id, target_edge, lane_len - 0.1)
                                except Exception:
                                    self.traci.vehicle.setStop(vehicle_id, target_edge, 0.0)
                            else:
                                self.traci.vehicle.setStop(vehicle_id, target_edge, stop_pos)
                            print(f"[dest_manager] set route to {target_edge} stop {stop_pos}")
                        else:
                            print(f"[dest_manager] could not compute route to {target_edge}; will retry later")
                            time.sleep(0.5)
                            q.put(new_dest_raw)
                    except Exception as e:
                        print(f"[dest_manager] error setting route to {target_edge}: {e}; will retry")
                        time.sleep(0.5)
                        q.put(new_dest_raw)

                except Exception as e:
                    print(f"[dest_manager] unexpected error processing new_dest '{new_dest}': {e}")
                    try:
                        q.put(new_dest_raw)
                    except Exception:
                        pass
                    time.sleep(0.5)
                finally:
                    q.task_done()

        
        # start watcher thread and destination processor thread
        dest_file = "new_dest.txt"
        # Ensure the file exists and is empty initially
        try:
            with open(dest_file, "w") as f:
                f.write("")
        except Exception:
            pass
        threading.Thread(target=watch_dest_file, args=(dest_file, dest_queue), daemon=True).start()

        try:
            vehicle_id = "0"
            depart_time = 0.0

            start_edge = "14"
            dest_edge  = "-14"
            # Compute a route automatically from start to destination
            route = traci.simulation.findRoute(start_edge, dest_edge)
            edge_list = route.edges

            # Add the vehicle to the network with that route
            traci.vehicle.add(vehicle_id, routeID="", depart=depart_time, departLane="best",
                            departPos="100.0", departSpeed="0")
            self.traci.vehicle.subscribe(vehicle_id, self.vehicleVariableList)
            traci.vehicle.setRoute(vehicle_id, edge_list)
            dest_length = traci.lane.getLength(dest_edge + "_0")
            # traci.vehicle.setStop(vehicle_id, dest_edge, dest_length-0.1)
            traci.vehicle.setStop(vehicle_id, dest_edge, 11)

            print(f"Added '{vehicle_id}'")

        except Exception as e:
            print(f"Error creating vehicle: {e}")

        # start the destination processing thread AFTER vehicle added
        threading.Thread(target=process_destinations, args=(vehicle_id, dest_queue), daemon=True).start()


        # create the SUMO display
        self.sumoDisplay = None
        if useDisplay:
            view = self.traci.gui.getIDList()[0]
            display = self.getDevice('sumo')
            if display is not None:
                from SumoDisplay import SumoDisplay # type: ignore
                self.sumoDisplay = SumoDisplay(display, displayZoom, view, directory, displayRefreshRate, displayFitSize,
                                               self.traci)


        # small helper to pick a random edge id from the net (filter out internal edges if needed)
        def random_edge_id():
            edges = [e.getID() for e in self.net.getEdges() if not e.getID().startswith(":")]
            return random.choice(edges)

        # create a random route between two distinct edges, returns list of edge ids or None
        def make_random_route(max_attempts=10):
            for _ in range(max_attempts):
                a = random_edge_id()
                b = random_edge_id()
                if a == b:
                    continue
                try:
                    r = self.traci.simulation.findRoute(a, b)
                    if r and len(r.edges) > 0:
                        return list(r.edges)
                except Exception:
                    pass
            return None

        # add (or re-add) a vehicle with given id and a random route
        def spawn_random_vehicle(vid=None, depart_delay=0.0):
            if vid is None:
                vid = "rnd-" + str(uuid.uuid4())[:8]
            route = make_random_route()
            if not route:
                # print(f"[traffic] could not compute random route for {vid}")
                return None
            try:
                # add vehicle (depart now). If id exists SUMO removed it, so re-adding is fine.
                self.traci.vehicle.add(vid, routeID="", depart=str(self.traci.simulation.getTime() + depart_delay),
                                    departLane="best", departPos="0.0", departSpeed="max")
                # then set the explicit route (some SUMO versions require setRoute after add)
                self.traci.vehicle.setRoute(vid, route)
                # optionally set a random type/speed
                # self.traci.vehicle.setSpeed(vid, random.uniform(5.0, 12.0))
                # print(f"[traffic] spawned {vid} route {route[:3]}...->{route[-1]}")
                return vid
            except Exception as e:
                # print(f"[traffic] failed to spawn {vid}: {e}")
                return None

        # how many background cars you want
        num_random_vehicles = 15 # traffic cars number
        background_vehicles = []

        # spawn initial set
        for i in range(num_random_vehicles):
            vid = f"bg{i}"
            v = spawn_random_vehicle(vid=vid, depart_delay=i * 0.2)  # slight stagger
            if v:
                background_vehicles.append(v)


        # Main simulation loop
        while self.step(step) >= 0:
            if self.usePlugin:
                sumoSupervisorPlugin.run(step)

            if self.sumoDisplay is not None:
                self.sumoDisplay.step(step)

            # try to perform a SUMO step, if it fails it means SUMO has been closed by the user
            try:
                self.traci.simulationStep()
            except self.traci.exceptions.FatalTraCIError:
                print("Sumo closed")
                self.sumoClosed = True
                break

            result = self.traci.simulation.getSubscriptionResults()

            # handle arrived vehicles reported by subscription (respawn them immediately)
            arrived = []
            if self.traci.constants.VAR_ARRIVED_VEHICLES_IDS in result:
                arrived = result[self.traci.constants.VAR_ARRIVED_VEHICLES_IDS]
                for vid in arrived:
                    # only respawn our background vehicles (optional filter)
                    if vid.startswith("bg") or vid.startswith("rnd-"):
                        # print(f"[traffic] vehicle arrived/finished: {vid} -> respawning")
                        # re-add same id immediately with a new route
                        spawn_random_vehicle(vid=vid, depart_delay=0.0)


            # SUMO simulation over (no more vehicle are expected)
            if result[self.traci.constants.VAR_MIN_EXPECTED_VEHICLES] == 0:
                break

            # subscribe to new vehicle
            for id in result[self.traci.constants.VAR_DEPARTED_VEHICLES_IDS]:
                if not id.startswith('webotsVehicle'):
                    self.traci.vehicle.subscribe(id, self.vehicleVariableList)
                elif self.sumoDisplay is not None and len(self.webotsVehicles) == 1:
                    # Only one vehicle controlled by Webots => center the view on it
                    self.traci.gui.trackVehicle(view, 'webotsVehicle0')


            # get result from the vehicle subscription and apply it
            idList = self.traci.vehicle.getIDList()
            for id in idList:
                self.get_vehicles_position(id, self.traci.vehicle.getSubscriptionResults(id),
                                           step, xOffset, yOffset, maximumLateralSpeed, maximumAngularSpeed,
                                           laneChangeDelay)
            self.disable_unused_vehicles(idList)

            # hide unused vehicles
            self.hide_unused_vehicles()

            if not disableTrafficLight:
                for id in self.trafficLights:
                    self.update_traffic_light_state(id, self.traci.trafficlight.getSubscriptionResults(id))

            self.update_vehicles_position_and_velocity(step, rotateWheels)
            self.update_webots_vehicles(xOffset, yOffset)

            # --- write speed & acceleration for vehicle "0" to text files (one value per file) ---
            try:
                # Only attempt if vehicle exists in SUMO
                vid = "0"
                id_list = self.traci.vehicle.getIDList()
                if vid in id_list:
                    try:
                        # traci returns speed in m/s, acceleration in m/s^2
                        speed_val = self.traci.vehicle.getSpeed(vid)
                    except Exception:
                        speed_val = float("nan")
                    try:
                        accel_val = self.traci.vehicle.getAcceleration(vid)
                    except Exception:
                        accel_val = float("nan")

                    # Optionally throttle writes to every N steps (set write_every = 1 to write every step)
                    write_every = 1
                    # use SUMO sim time to reduce dependency on step counts (optional)
                    current_step = int(self.traci.simulation.getTime() / (step / 1000.0)) if step > 0 else 0
                    if write_every <= 1 or (current_step % write_every == 0):
                        try:
                            # overwrite the files with single-line floats (easy to parse)
                            with open(speed_file, "w") as f:
                                f.write(f"{speed_val:.6f}\n")
                        except Exception as e:
                            print(f"Error writing speed file: {e}")
                        try:
                            with open(accel_file, "w") as f:
                                f.write(f"{accel_val:.6f}\n")
                        except Exception as e:
                            print(f"Error writing acceleration file: {e}")
                else:
                    # vehicle not present; write NaN or keep previous values (choose behavior)
                    with open(speed_file, "w") as f:
                        f.write("NaN\n")
                    with open(accel_file, "w") as f:
                        f.write("NaN\n")
            except Exception as e:
                # keep the loop alive on error
                print(f"[speed/accel writer] error: {e}")
            # --- end speed/accel writer ---


        if not self.sumoClosed:
            self.traci.close()
        else:
            self.stop_all_vehicles()
        sys.stdout.flush()
