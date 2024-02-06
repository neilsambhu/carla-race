import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
import queue
from tensorflow.keras.applications.xception import Xception # Neil modified `from keras.applications.xception import Xception`
from tensorflow.keras.layers import Concatenate, Dense, GlobalAveragePooling2D # Neil modified `from keras.layers import Dense, GlobalAveragePooling2D`
from tensorflow.keras.optimizers import Adam # Neil modified `from keras.optimizers import Adam`
from tensorflow.keras.models import Model # Neil modified `from keras.models import Model`

import tensorflow as tf
import tensorflow.keras.backend as backend # Neil modified `import keras.backend.tensorflow_backend as backend`
# from threading import Thread
# from tensorflow.keras.callbacks import TensorBoard # Neil modified `from keras.callbacks import TensorBoard`

from tqdm import tqdm
import pickle
import subprocess
import statistics

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla, configparser

config = configparser.ConfigParser()
config.read('config.ini')
bSAMBHU24 = config.getboolean('Settings','bSAMBHU24')
bA100 = config.getboolean('Settings','bA100')
bGPU_random = config.getboolean('Settings','bGPU_random')
bGPU13 = config.getboolean('Settings','bGPU13')
bGPU43 = config.getboolean('Settings','bGPU43')
bGPU45 = config.getboolean('Settings','bGPU45')
bGPU46 = config.getboolean('Settings','bGPU46')
bGPU47 = config.getboolean('Settings','bGPU47')
bGAIVI = not bSAMBHU24

SHOW_PREVIEW = False
# IM_WIDTH = 600
IM_WIDTH = 128
# IM_HEIGHT = 600
IM_HEIGHT = 128
# SECONDS_PER_EPISODE = 10
SECONDS_PER_EPISODE = 3*60
# REPLAY_MEMORY_SIZE = 5_000
# MIN_REPLAY_MEMORY_SIZE = 1_000
# MIN_REPLAY_MEMORY_SIZE = int(1.5*SECONDS_PER_EPISODE*20) # 12/24/2023 6:37 AM: Neil commented out
path_AP_locations = '_out_07CARLA_AP/Locations_Town04_0_335.txt'
with open(path_AP_locations, 'r') as file:
    COUNT_LOCATIONS = len(file.readlines())
MIN_REPLAY_MEMORY_SIZE = COUNT_LOCATIONS
# MIN_REPLAY_MEMORY_SIZE = int(64 * COUNT_LOCATIONS)
# REPLAY_MEMORY_SIZE = 5*COUNT_LOCATIONS
# REPLAY_MEMORY_SIZE = 50_000
# REPLAY_MEMORY_SIZE = COUNT_LOCATIONS
COUNT_FRAME_WINDOW = 10*20
# COUNT_FRAME_WINDOW = 5
REPLAY_MEMORY_SIZE = COUNT_LOCATIONS + COUNT_FRAME_WINDOW - 1
MINIBATCH_SIZE = None
if bSAMBHU24:
    MINIBATCH_SIZE = 50_000//65536
else:
    if bGPU_random:
        MINIBATCH_SIZE = 1000 # 8 minutes per epoch
    elif bGPU13:
        MINIBATCH_SIZE = 1000 # 7 minutes per epoch; with 1 GPU, 8 minutes per epoch
    elif bGPU43:
        # GPU45
        # MINIBATCH_SIZE = REPLAY_MEMORY_SIZE // 8192
        # MINIBATCH_SIZE = 1
        # MINIBATCH_SIZE = REPLAY_MEMORY_SIZE // 1024 #failure
        # MINIBATCH_SIZE = REPLAY_MEMORY_SIZE // 4096 # =12->3 3600 seconds/epoch
        MINIBATCH_SIZE = REPLAY_MEMORY_SIZE // 2048 #first batch trained
        MINIBATCH_SIZE = 36 #first batch trained
        MINIBATCH_SIZE = 40 #first batch trained
        MINIBATCH_SIZE = 44 #first batch trained
        MINIBATCH_SIZE = 46 #first batch trained
        MINIBATCH_SIZE = 100 #failure
        MINIBATCH_SIZE = 48 #first batch trained; warning
        MINIBATCH_SIZE = 75 #first batch trained; warning
        MINIBATCH_SIZE = 87 #first batch trained; warning
        MINIBATCH_SIZE = 93 #failure
        # MINIBATCH_SIZE = 96 #failure

        # batch size decrease by factor of 4
        MINIBATCH_SIZE = 22 # not tried
        MINIBATCH_SIZE = 3*11 #failure
        MINIBATCH_SIZE = 3*5 # 3685 seconds/epoch

        # GPU43
        MINIBATCH_SIZE = 4*8 
        MINIBATCH_SIZE = 4*64
        MINIBATCH_SIZE = 4*512
        MINIBATCH_SIZE = REPLAY_MEMORY_SIZE
        
        # 1 GPU
        MINIBATCH_SIZE = 16 # 1 hour 8 minutes per epoch
        # 4 GPUs
        MINIBATCH_SIZE = 4*16 # failure
        MINIBATCH_SIZE = 4*12 # failure
        MINIBATCH_SIZE = 4*8 # failure
        MINIBATCH_SIZE = 4*4 # 1 hour 10 minutes per epoch
        MINIBATCH_SIZE = REPLAY_MEMORY_SIZE - COUNT_FRAME_WINDOW
        MINIBATCH_SIZE = 1000 # failure
        MINIBATCH_SIZE = 100 # 1 hour 8 minutes per epoch (training batch size 1)
    elif bGPU45:
        MINIBATCH_SIZE = 8 # 1 hour 3 minutes per epoch
        MINIBATCH_SIZE = 64 # failure
        MINIBATCH_SIZE = 48 # failure
        MINIBATCH_SIZE = 16 # 56 minutes per epoch
        MINIBATCH_SIZE = REPLAY_MEMORY_SIZE - COUNT_FRAME_WINDOW # not useful; main memory not big enough
        MINIBATCH_SIZE = 32 # failure
        MINIBATCH_SIZE = 24 # failure
        MINIBATCH_SIZE = 20 # 56 minutes per epoch
        MINIBATCH_SIZE = 20 # 51 minutes per epoch
        # MINIBATCH_SIZE = 100 # 47 minutes per epoch (training batch size 20) # failure
        # MINIBATCH_SIZE = 200 # 56 minutes per epoch (training batch size 1)
        # MINIBATCH_SIZE = 200 # 47 minutes per epoch (training batch size 20)
        # MINIBATCH_SIZE = 400 # 49 minutes per epoch (training batch size 20) # failure

        # 5-frame lookback
        MINIBATCH_SIZE = 20 # 32 minutes per epoch
        MINIBATCH_SIZE = REPLAY_MEMORY_SIZE - COUNT_FRAME_WINDOW
        MINIBATCH_SIZE = 200 # 7 minutes per epoch
        MINIBATCH_SIZE = 2000 # failure
        MINIBATCH_SIZE = 1000 # 5 minutes per epoch
        MINIBATCH_SIZE = 1500 # 5 minutes per epoch

        # 4-layer, 28 filters
        MINIBATCH_SIZE = (REPLAY_MEMORY_SIZE - COUNT_FRAME_WINDOW)//128 # 95 seconds per epoch; 64-node LSTM
        MINIBATCH_SIZE = COUNT_LOCATIONS # failure
        MINIBATCH_SIZE = COUNT_LOCATIONS//4
        MINIBATCH_SIZE = COUNT_LOCATIONS//16
        MINIBATCH_SIZE = COUNT_LOCATIONS//64 # 90 seconds per epoch; 64-node LSTM

        # 8-node LSTM
        MINIBATCH_SIZE = COUNT_LOCATIONS//64 # 90 seconds per epoch
        # MINIBATCH_SIZE = COUNT_LOCATIONS//16 # failure

        # 4 layers, no pooling
        MINIBATCH_SIZE = 1
    elif bGPU46:
        MINIBATCH_SIZE = 1500 # 5 minutes per epoch
    elif bGPU47:
        MINIBATCH_SIZE = 20 # 51 minutes per epoch

# MIN_REPLAY_MEMORY_SIZE = 30_000
# MIN_REPLAY_MEMORY_SIZE = MINIBATCH_SIZE + COUNT_FRAME_WINDOW - 1
# MIN_REPLAY_MEMORY_SIZE = max(COUNT_FRAME_WINDOW, MINIBATCH_SIZE)
MIN_REPLAY_MEMORY_SIZE = MINIBATCH_SIZE
PREDICTION_BATCH_SIZE = 1
# TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
TRAINING_BATCH_SIZE = MINIBATCH_SIZE
# TRAINING_BATCH_SIZE = 20
# TRAINING_BATCH_SIZE = 16
# TRAINING_BATCH_SIZE = 1
UPDATE_TARGET_EVERY = 5
# MODEL_NAME = "Xception"
MODEL_NAME = "Neil_SDC_2023"

MEMORY_FRACTION = 0.8
MIN_REWARD = -200

# EPISODES = 100
# EPISODES = 5
# EPISODES = 1000
EPISODES = 10_000

DISCOUNT = 0.99
epsilon_base = 1.0
epsilon = 1.0
EPSILON_DECAY = 0.95
# EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


directory_output = '_out_16rl_custom2'
# if os.path.exists(directory):
#     [os.remove(os.path.join(directory, file)) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
# else:
#     print("Directory does not exist or is already removed.")
bSync = True
bVerbose = True
bGPU = True

# Define action space
# action_space = {'throttle': np.linspace(0.0, 1.0, num=11),
# action_space = {'throttle': np.linspace(0.0, 1.0, num=2),
#                 # 'steer': np.linspace(-1.0, 1.0, num=21),
#                 'steer': np.linspace(-1.0, 1.0, num=3),
#                 # 'brake': np.linspace(0.0, 1.0, num=11)}
#                 # 'brake': np.linspace(0.0, 0.0, num=11)}
#                 # 'brake': np.linspace(0.0, 1.0, num=2)}
#                 'brake': np.linspace(0.0, 0.0, num=1)}
action_space = {'brake_throttle': np.linspace(-1.0, 1.0, num=3),
                'steer': np.linspace(-1.0, 1.0, num=3)}
# print(action_space);import sys;sys.exit()
# action_size = len(action_space['throttle'])*len(action_space['steer'])*len(action_space['brake'])
action_size = len(action_space['brake_throttle'])*len(action_space['steer'])


if bGPU:
    if not bGAIVI:
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f'GPUs: {gpus}')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    class CarEnv:
        SHOW_CAM = SHOW_PREVIEW
        STEER_AMT = 1.0
        im_width = IM_WIDTH
        im_height = IM_HEIGHT
        front_camera = None
        episode = None
        action_space = action_space
        idx_tick = -1
        pathImage = ''
        queueImagesWritten = queue.Queue()

        def __init__(self):
            # self.client = carla.Client("localhost", 2000)
            if not bGAIVI:
                self.client = carla.Client("10.247.52.30", 2000)
                self.client.set_timeout(600)
            else:
                command_output = subprocess.run(['squeue'], capture_output=True, text=True)
                output_lines = command_output.stdout.split('\n')
                carla_line = [line for line in output_lines if 'nsambhu' in line and 'carla.sh' in line and 'GPU' in line]
                gpu_info = carla_line[-1].split()[-1]  # Assuming GPU info is the last column
                print("GPU Info for carla.sh:", gpu_info)
                self.client = carla.Client(gpu_info, 2000)
                self.client.set_timeout(120)
            # self.world = self.client.get_world()
            self.world = self.client.load_world('Town04_Opt')
            self.client.set_timeout(2.0) # 12/19/2023 11:45 PM: Neil added
            # self.client.set_timeout(60)
            self.blueprint_library = self.world.get_blueprint_library()
            self.model_3 = self.blueprint_library.filter("model3")[0]
            if bSync:
                # print(f'bSync set synchronous mode')
                # Set synchronous mode 
                settings = self.world.get_settings()
                settings.synchronous_mode = True # Enables synchronous mode
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)
                traffic_manager = self.client.get_trafficmanager()
                traffic_manager.set_synchronous_mode(True)
                # Reload world
                # self.client.reload_world()

        def reset(self):
            self.collision_hist = []
            self.actor_list = []
            self.idx_tick = -1
            self.pathImage = ''
            self.queueImagesWritten = queue.Queue()

            # self.transform = random.choice(self.world.get_map().get_spawn_points())
            self.transform = self.world.get_map().get_spawn_points()[0]
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            self.actor_list.append(self.vehicle)
            if bSync:
                # print('bSync reset: spawn actor')
                self.world.tick()
                self.idx_tick += 1

            self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
            self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
            self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
            self.rgb_cam.set_attribute("fov", f"110")

            # transform = carla.Transform(carla.Location(x=2.5, z=0.7)) 
            # revision: 
            transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
            self.actor_list.append(self.sensor)
            self.sensor.listen(lambda data: self.process_img(data))

            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
            if not bSync:
                time.sleep(4)
            elif bSync:
                # print('bSync reset: apply control')
                self.world.tick()
                self.idx_tick += 1

            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event))

            while self.front_camera is None:
                time.sleep(0.01)

            # self.episode_start = time.time()
            self.episode_start = self.world.get_snapshot().timestamp.elapsed_seconds
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

            if bSync and False:
                self.world.tick()
                self.idx_tick += 1

            return self.front_camera

        def collision_data(self, event):
            self.collision_hist.append(event)

        def process_img(self, image):
            # print(image.height, image.width)
            i = np.array(image.raw_data)
            # print(i.shape)
            i2 = i.reshape((self.im_height, self.im_width, 4))
            # i3 = i2[:, :, :3]
            i3 = cv2.cvtColor(i2, cv2.COLOR_BGRA2RGB)
            # print('type(i3)',type(i3))
            if self.SHOW_CAM:
                cv2.imshow("", i3)
                cv2.waitKey(1)
            self.front_camera = i3
            from PIL import Image
            i4 = Image.fromarray(i3)
            if not os.path.exists('%s/%04d' % (directory_output, self.episode)):
                os.makedirs('%s/%04d' % (directory_output, self.episode))
                time.sleep(1)
            self.pathImage = '%s/%04d' % (directory_output, self.episode)
            i4.save('%s/%04d/%06d.png' % (directory_output, self.episode, image.frame))
            # i4.save('%s/%04d/%06d.jpg' % (directory_output, self.episode, image.frame))
            # count_checkFileExists = 0
            while not os.path.exists('%s/%04d/%06d.png' % (directory_output, self.episode, image.frame)):
                # count_checkFileExists += 1
                time.sleep(0.1)
            self.queueImagesWritten.put(self.pathImage)
            # print(f'count_checkFileExists: {count_checkFileExists}')
            # self.queueImagesWritten.get()
            # if self.done:
            #     bFinalImageWritten = True


        def step(self, action):
            if bVerbose and False:
                print(f'action: {action}')
                print(f'action[0]: {action[0]}')
                print(f'type(action[0]): {type(action[0])}')
            brake_throttle_index, steer_index = np.unravel_index(action, (len(action_space['brake_throttle']), len(action_space['steer'])))

            selected_brake_throttle = action_space['brake_throttle'][brake_throttle_index]
            selected_steer = action_space['steer'][steer_index]

            throttle_value, steer_value, brake_value = selected_brake_throttle, selected_steer, selected_brake_throttle
            if selected_brake_throttle < 0:
                throttle_value = 0.0
                brake_value = -1*selected_brake_throttle
            if selected_brake_throttle > 0:
                brake_value = 0.0

            if bVerbose and False:
                print(f'action: {action}\tthrottle: {throttle_value}\tsteer: {steer_value}\tbrake: {brake_value}')

            self.vehicle.apply_control(
                carla.VehicleControl(throttle=float(throttle_value), steer=float(steer_value), brake=float(brake_value))
            )

            if bSync:
                # print(f'bSync step: after applying vehicle control')
                self.world.tick() # Neil added
                self.idx_tick += 1

            done = False
            reward = 0
            lines = []
            location_groundTruth = carla.Location(0.0,0.0,0.0)
            # Reading ground truth coordinates from the file
            with open(path_AP_locations, 'r') as file:
                lines = file.readlines()
                if self.idx_tick < len(lines):
                    data = lines[self.idx_tick].split()
                    if len(data) >= 3:
                        location_groundTruth = carla.Location(float(data[0]), float(data[1]), float(data[2]))  # Extracting x, y, z coordinates

            # Get the current location of the vehicle
            location_current = self.vehicle.get_location()
            carla_location_current = carla.Location(location_current.x, location_current.y, location_current.z)

            # Calculate the distance between the vehicle's current location and ground truth location
            distance = location_groundTruth.distance(carla_location_current)
            # You might want to define a threshold and reward scheme based on the distance
            # For example, if distance < threshold: reward = some_value
            # Modify the reward calculation based on your requirements
            def getRewardDistance():
                # reward = 0
                # reward = -1*distance**3 - distance + 1
                # reward = -1*distance**3 - distance + 100
                reward = 10 - distance 
                # if distance < 10:
                #     reward += 10 - distance
                # else:
                #     reward -= 10
                return max(reward, -10)
            # reward += getRewardDistance()
            reward -= distance

            # Set 'done' flag to True when ticks exceed the lines in the file
            done = self.idx_tick >= len(lines)
            # done = self.idx_tick >= 100

            # v = self.vehicle.get_velocity()
            # kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
            # if kmh <= 15:
            #     reward += kmh            
            # elif kmh > 15:
            #     reward += 15
            # if kmh > 0:
            #     reward += getRewardDistance()

            # if self.episode_start + SECONDS_PER_EPISODE < time.time():
            # if self.episode_start + SECONDS_PER_EPISODE < self.world.get_snapshot().timestamp.elapsed_seconds:
            #     done = True

            if len (self.collision_hist) != 0:
                done = True
                # reward = -200
                # reward = -0.001
                # reward = -1

            return self.front_camera, reward, done, None
    
        def get_count_vehicles(self):
            actors = self.world.get_actors()
            vehicles = [actor for actor in actors if actor.type_id.startswith('vehicle')]
            return len(vehicles)

    class DQNAgent:
        def __init__(self):
            self.model = self.create_model()
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())
            self.saved_model = self.create_model()
            self.saved_model.set_weights(self.model.get_weights())
            self.count_saved_models = 0
            self.count_batches_trained = 0
            self.count_epochs_trained = 0

            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

            self.target_update_counter = 0
            # Neil commented `self.graph = tf.get_default_graph()`

            self.terminate = False
            self.last_logged_episode = 0
            self.training_initialized = False

        def create_model(self):
            from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, AveragePooling2D, MaxPooling2D, TimeDistributed, LSTM, Bidirectional
            from tensorflow.keras.models import Model
            input_shape = (COUNT_FRAME_WINDOW, IM_HEIGHT, IM_WIDTH, 3)
            # count_filters = 1 # 2/4/2024 2:47 AM: 73 seconds per epoch
            count_filters = 28
            pool_size = (2,2) # 2/4/2024 2:57 AM: 53 seconds per epoch

            # Define the input layer
            input_layer = Input(shape=input_shape)

            base_model = None
            for i in range(0,4):
                if i == 0:
                    base_model = TimeDistributed(Conv2D(count_filters, (3,3), padding='same'))(input_layer) # 2/4/2024 2:52 AM: 55 seconds per epoch
                else:
                    base_model = TimeDistributed(Conv2D(count_filters, (3,3), padding='same'))(base_model)
                base_model = TimeDistributed(BatchNormalization())(base_model)
                base_model = TimeDistributed(Activation('relu'))(base_model)
            base_model = TimeDistributed(MaxPooling2D(pool_size=pool_size))(base_model)

            x = TimeDistributed(Flatten())(base_model)
            # x = Flatten()(base_model)
            # x = Flatten()(x)
            print(f'x.shape after flatten: {x.shape}')


            # Apply LSTM layer
            x = Bidirectional(LSTM(units=1024, return_sequences=True))(x)
            x = Bidirectional(LSTM(units=1024, return_sequences=False))(x) # 2/5/2024 11:28 AM: 111 seconds per epoch
            # x = LSTM(units=1024)(x) # 3.5 minutes per epoch
            # x = LSTM(units=64)(x) # 2/4/2024 12:35 AM: 95 seconds per epoch
            # x = LSTM(units=128)(x) # 5 seconds per epoch
            # x = LSTM(units=512)(x) # 2/3/2024 11:34 PM: 100 seconds per epoch
            # x = LSTM(units=1024)(x) # 90 seconds per epoch
            # x = LSTM(units=4096)(x) # 215 seconds per epoch
            # x = LSTM(units=8)(x) # 2/4/2024 2:42 AM: 90 seconds per epoch
            
            # print(f'x.shape after LSTM: {x.shape}')
            size_reduce = 2
            while x.shape.as_list()[1] >= size_reduce * (action_size + 1):
                # x = TimeDistributed(Dense(x.shape.as_list()[2] // size_reduce, activation="relu"))(x)
                x = Dense(x.shape.as_list()[1] // size_reduce, activation="relu")(x)

            # Define the output layer
            # output_layer = TimeDistributed(Dense(action_size, activation="linear"))(x)
            output_layer = Dense(action_size, activation="linear")(x)

            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
            # print(model.summary())
            return model

        def update_replay_memory(self, transition):
            # transition = (current_state, action, reward, new_state, done)
            self.replay_memory.append(transition)

        def train(self, epochs):
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return

            # minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
            # intRangeToSample = len(self.replay_memory) - COUNT_FRAME_WINDOW + 1
            intRangeToSample = len(self.replay_memory)
            if bVerbose and False:
                print(f'len(self.replay_memory): {len(self.replay_memory)}\tintRangeToSample: {intRangeToSample}\tMINIBATCH_SIZE: {MINIBATCH_SIZE}')
                time.sleep(10)
                print(f'len(self.replay_memory): {len(self.replay_memory)}\tintRangeToSample: {intRangeToSample}\tMINIBATCH_SIZE: {MINIBATCH_SIZE}')
            sampled_indices = random.sample(range(0, intRangeToSample), min(intRangeToSample, MINIBATCH_SIZE))
            # sampled_indices = random.sample(range(0, intRangeToSample), MINIBATCH_SIZE)
            # sampled_indices = random.sample(range(COUNT_FRAME_WINDOW-1, len(self.replay_memory)), min(intRangeToSample, MINIBATCH_SIZE))
            if bVerbose and False:
                print(f'sampled_indices: {sampled_indices}')
            minibatch = []
            # for index in sampled_indices:
            for index in range(0,intRangeToSample):
                sequence = list(self.replay_memory)[index:index + COUNT_FRAME_WINDOW]
                minibatch.append(sequence)

            # current_states = np.array([transition[0] for transition in minibatch])/255
            # current_states = np.array([transition[0] for transition in minibatch])
            # current_states = np.array([current_state for transition[0] in transition in minibatch])
            black_image = np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
            current_states = []
            new_current_states = []
            for sequence in minibatch:
                window_current_states = [np.copy(black_image) for _ in range(COUNT_FRAME_WINDOW)]
                window_new_current_states = [np.copy(black_image) for _ in range(COUNT_FRAME_WINDOW)]
                bFound_done = False
                frame_0_last = None
                frame_3_last = None
                for frame in sequence:
                    if frame[4]:
                        if bVerbose and False:
                            print("found final frame in drive")
                        bFound_done = True
                        frame_0_last = frame[0]
                        frame_3_last = frame[3]
                    if not bFound_done:
                        window_current_states.append(frame[0])
                        window_new_current_states.append(frame[3])
                    else:
                        window_current_states.append(frame_0_last)
                        window_new_current_states.append(frame_3_last)
                        break
                current_states.append(window_current_states[-COUNT_FRAME_WINDOW:])
                new_current_states.append(window_new_current_states[-COUNT_FRAME_WINDOW:])
            current_states = np.asarray(current_states)
            new_current_states = np.asarray(new_current_states)
            if bVerbose and False:
                print(f'len(minibatch): {len(minibatch)}\tcurrent_states.shape: {current_states.shape}\tnew_current_states.shape: {new_current_states.shape}')
            # Neil commented `with self.graph.as_default():`
            # current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE) # Neil left tabbed 1
            # current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE, verbose=0)
            current_qs_list = None
            try:
                current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE, verbose=0)
            except Exception as e:
                print(f'Error message: {e}')
            
            # new_current_states = np.array([transition[3] for transition in minibatch])/255
            # new_current_states = np.array([transition[3] for transition in minibatch])
            # new_current_states = np.array([transition[:][3] for transition in minibatch])
            # Neil commented `with self.graph.as_default():`
            # future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE) # Neil left tabbed 1
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE, verbose=0)

            x = []
            y = []

            for index, sequence in enumerate(minibatch):
                current_state, action, reward, new_state, done = sequence[-1]
                if bVerbose and False:
                    print(f'action: {action}')
                if bVerbose and False: 
                    if len(sequence) is not COUNT_FRAME_WINDOW:
                        print(f'len(sequence): {len(sequence)}')
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                if bVerbose and False:
                    print(f'current_qs_list: {current_qs_list}')
                    print(f'new_q: {new_q}')
                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                # window_x = []
                # # for frame in sequence:
                # for frame in current_states[index]:
                #     if bVerbose:
                #         # print(f'type(frame[0]: {type(frame[0])}')
                #         if frame[0].shape != (128, 128, 3):
                #             print(f'frame[0].shape: {frame[0].shape}')
                #         if frame[0].dtype != 'uint8':
                #             print(f'frame[0].dtype: {frame[0].dtype}')
                #     window_x.append(frame[0])
                # x.append(window_x)
                x.append(current_states[index])
                y.append(current_qs)

            log_this_step = False

            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, verbose=1, start_from_epoch=1)
            # callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', baseline=1.0)
            hist = self.model.fit(
                np.array(x),
                np.array(y),
                batch_size=TRAINING_BATCH_SIZE,
                epochs=epochs,
                # verbose=0,
                # verbose=1,
                # verbose='auto',
                verbose=2,
                shuffle=False,
                # callbacks=[self.tensorboard] if log_this_step else None # 12/18/2023 7:47 PM: Neil commented out
                callbacks=[callback]
            )

            if log_this_step:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

            if self.count_batches_trained == 0:
                print('Finished training first batch.')
            self.count_batches_trained += 1
            self.saved_model.set_weights(self.model.get_weights())
            self.count_saved_models += 1

            return hist.history

        def get_qs(self, state):
            return self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

if __name__ == "__main__":
    FPS = 60
    ep_rewards = [-200]
    # ep_rewards = [-0.0001]

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1) # Neil modified `tf.set_random_seed(1)`

    agent = DQNAgent()
    x = np.random.uniform(size=(1, COUNT_FRAME_WINDOW, IM_HEIGHT, IM_WIDTH, 3)).astype(np.uint8)
    y = np.random.uniform(size=(1, action_size)).astype(np.ushort)
    agent.model.fit(x, y, verbose=False, batch_size=1) # Neil left tabbed 1

    idx_episode_start = 1
    idx_action1 = 0
    idx_action2 = 0
    max_framesPerAction = 100
    count_framesPerAction1 = 0
    count_framesPerAction2 = 0
    import glob, shutil
    bLoadReplayMemory = False
    episodeToRecover = '0008'
    if bLoadReplayMemory:        
        with open(f'bak/{episodeToRecover}.replay_memory', 'rb') as file:
            agent.replay_memory = pickle.load(file)
        idx_episode_start = int(episodeToRecover)+1
        idx_action1 = 2530+1
        epsilon = epsilon_base*(EPSILON_DECAY**int(episodeToRecover))
        epsilon = max(MIN_EPSILON, epsilon)
    bLoadModel = False
    if bLoadModel:
        agent.model = tf.keras.models.load_model(glob.glob(f'bak/{episodeToRecover}.*.model')[0])
    matching_files = glob.glob(os.path.join('tmp', '*.model'))
    if len(matching_files) > 0:
        matching_files.sort()
        print(f'Models in tmp {matching_files}')
        print(f'Load model {matching_files[-1]}')
        agent.model = tf.keras.models.load_model(matching_files[-1])

        idx_episode_saved, agent.count_epochs_trained = matching_files[-1].split('/')[1].split('.')[0:2]
        idx_episode_saved, agent.count_epochs_trained = int(idx_episode_saved), int(agent.count_epochs_trained)
        idx_episode_crashed = idx_episode_saved + 1
        # remove leftover images from failed episode
        matching_files = glob.glob(os.path.join(directory_output, f'*{idx_episode_crashed}', '*.png'))
        matching_files.sort()
        print(f'Leftover images from failed episode: {matching_files}')

        matching_files = glob.glob(os.path.join('tmp', '*.replay_memory'))
        matching_files.sort()
        print(f'Replay memory in tmp {matching_files}')
        print(f'Load replay memory {matching_files[-1]}')
        with open(matching_files[-1], 'rb') as file:
            agent.replay_memory = pickle.load(file)

        matching_files = glob.glob(os.path.join('tmp', '*.idx_action1'))
        if len(matching_files)>0:
            matching_files.sort()
            print(f'Index action in temp {matching_files}')
            print(f'load idx_action1 {matching_files[-1]}')
            idx_action1 = matching_files[-1].split('/')[1].split('.')[0]
            idx_action1 = int(idx_action1)

        idx_episode_start = idx_episode_crashed

    env = CarEnv()

    # trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    # trainer_thread.start()

    # while not agent.training_initialized:
    #     time.sleep(0.01)

    bTrainingComplete = False
    previousEpisode_countBatchesTrained = agent.count_batches_trained
    bAction2Started = False
    bAction2Finished = False
    try:
        for episode in tqdm(range(idx_episode_start, EPISODES+1), ascii=True, unit="episode"):
            # time.sleep(random.uniform(0,60))
            # count_vehicles = env.get_count_vehicles()
            # print(f'episode: {episode}\tcount_vehicles: {count_vehicles}')
            # if count_vehicles == 0:
            if True:
                lookback = 2
                if episode > lookback:
                    matching_files = glob.glob(os.path.join('tmp', f'*{episode-lookback}.*.model'))
                    [shutil.rmtree(matching_file) for matching_file in matching_files]
                    matching_files = glob.glob(os.path.join('tmp', f'*{episode-lookback}.replay_memory'))
                    [os.remove(matching_file) for matching_file in matching_files]
                lookback = 200
                if episode > lookback:
                    matching_files = glob.glob(os.path.join(directory_output, f'*{episode-lookback}'))
                    [shutil.rmtree(matching_file) for matching_file in matching_files]

                print(f'\nStarted episode {episode} of {EPISODES}')
                if bGAIVI:
                    nvidia_smi = subprocess.Popen('nvidia-smi', shell=True, preexec_fn=os.setsid)

                env.collision_hist = []
                # agent.tensorboard.step = episode
                env.episode = episode
                episode_reward = 0
                # step = 1
                current_state = env.reset()
                window_current_state = deque(maxlen=COUNT_FRAME_WINDOW)
                black_image = np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
                for i in range(COUNT_FRAME_WINDOW-1):
                    window_current_state.append(np.asarray(black_image))
                window_current_state.append(np.asarray(current_state))

                if bSync and False:
                    env.world.tick()
                    env.idx_tick += 1
                done = False
                idx_control = 0
                # for i in range(0,10):
                #     env.world.tick()
                count_frames_completed = 0

                count_action_model = 0
                action_random = np.random.randint(0, action_size) # fine to have this duplicated
                count_action_random = 0
                max_count_action = 1*20
                while True:
                    if bSync and False:
                        # print(f'bSync inside episode')
                        env.world.tick();
                        env.idx_tick += 1
                    action = None
                    if (np.random.random() > epsilon and count_action_random == 0) or (count_action_model > 0):
                    # if len(agent.replay_memory) < REPLAY_MEMORY_SIZE:
                    # if False:
                        # action = np.argmax(agent.get_qs(current_state))
                        action = np.argmax(agent.get_qs(np.asarray(window_current_state)))                    
                        count_action_model += 1
                        if count_action_model > max_count_action:
                            count_action_model = 0
                        # with open('_out_07CARLA_AP/Controls_Town04_0_335.txt', 'r') as file:
                        #     lines = file.readlines()
                        #     throttle,steer,brake = lines[idx_control].split()
                        #     idx_control += 1
                        #     throttle,steer,brake = float(throttle),float(steer),float(brake)
                        #     throttle_index = np.abs(action_space['throttle'] - throttle).argmin()
                        #     steer_index = np.abs(action_space['steer'] - steer).argmin()
                        #     brake_index = np.abs(action_space['brake'] - brake).argmin()
                        #     action = throttle_index * len(action_space['steer']) * len(action_space['brake']) + \
                        #                 steer_index * len(action_space['brake']) + \
                        #                 brake_index
                    else:
                        # bAction1Valid = False
                        # while not bAction1Valid:
                        #     action = np.random.randint(0, action_size)
                        #     throttle_action = action // (len(action_space['steer'])*len(action_space['brake']))
                        #     brake_action = action % len(action_space['brake'])
                        #     if throttle_action == 0 or brake_action == 0:
                        #         bAction1Valid = True
                        if count_action_random > max_count_action:
                            action_random = np.random.randint(0, action_size)
                            count_action_random = 0
                        action = action_random
                        count_action_random += 1
                        # action = np.argmax(agent.get_qs(np.asarray(window_current_state)))                    
                        if not bSync:
                            time.sleep(1/FPS)
                    # if idx_action1 < action_size:
                    #     count_framesPerAction1 += 1
                    #     if count_framesPerAction1 > max_framesPerAction:
                    #         print(f'Finished idx_action1: {idx_action1}\taction size: {action_size}')
                    #         # # transition to second action
                    #         # count_framesPerAction2 += 1
                    #         # if count_framesPerAction2 > max_framesPerAction:
                    #         #     bAction2Finished = True
                    #         #     print(f'Finished idx_action2: {idx_action2}\taction size: {action_size}')
                    #         # if bAction2Finished:
                    #         #     idx_action2 += 1
                    #         #     count_framesPerAction2 = 0
                    #         #     bAction2Started = False
                    #         # if not bAction2Started:
                    #         #     idx_action2 = 0
                    #         #     count_framesPerAction2 = 0
                    #         #     bAction2Started = True
                    #         # bAction2Valid = False
                    #         # while not bAction2Valid and idx_action2 < action_size:
                    #         #     action = idx_action2
                    #         #     throttle_action = action // (len(action_space['steer'])*len(action_space['brake']))
                    #         #     brake_action = action % len(action_space['brake'])
                    #         #     if brake_action == 0 and throttle_action > 0 and idx_action2 != idx_action1:
                    #         #         bAction2Valid = True
                    #         #         matching_files = glob.glob(os.path.join('tmp', '*idx_action2'))
                    #         #         [os.remove(matching_file) for matching_file in matching_files]
                    #         #         open(f'tmp/{idx_action2:04d}.idx_action2', "w")
                    #         #     else:
                    #         #         idx_action2 += 1
                    #         idx_action1 += 1
                    #         count_framesPerAction1 = 1

                            
                    #     bAction1Valid = False
                    #     while not bAction1Valid and idx_action1 < action_size:
                    #         action = idx_action1
                    #         throttle_action = action // (len(action_space['steer'])*len(action_space['brake']))
                    #         brake_action = action % len(action_space['brake'])
                    #         if brake_action == 0 and throttle_action > 0:
                    #             bAction1Valid = True
                    #             matching_files = glob.glob(os.path.join('tmp', '*idx_action1'))
                    #             [os.remove(matching_file) for matching_file in matching_files]
                    #             open(f'tmp/{idx_action1:04d}.idx_action1', "w")
                    #         else:
                    #             idx_action1 += 1
                    #     if idx_action1 >= action_size:
                    #         print(f'Finished initializing all actions. Predicting from model.')
                    #         # action = np.argmax(agent.get_qs(current_state))                    
                    #         action = np.argmax(agent.get_qs(np.asarray(window_current_state)))                    
                    # else:
                    #     # action = np.argmax(agent.get_qs(current_state))                    
                    #     action = np.argmax(agent.get_qs(np.asarray(window_current_state)))                    

                    new_state, reward, done, _ = env.step(action)            
                    count_frames_completed += 1
                    episode_reward += reward
                    agent.update_replay_memory((current_state, action, reward, new_state, done))
                    window_current_state.append(np.asarray(current_state))
                    # from PIL import Image
                    # i4 = Image.fromarray(current_state)
                    # if not os.path.exists('%s/%04d' % (directory_output, episode)):
                    #     os.makedirs('%s/%04d' % (directory_output, episode))
                    #     time.sleep(1)
                    # # i4.save('%s/%04d/%06d.png' % (directory_output, self.episode, image.frame))
                    # i4.save('%s/%04d/%06d.jpg' % (directory_output, self.episode, image.frame))                

                    if done:
                        env.done = True
                        break

                # while env.queueImagesWritten.qsize() > 0:
                # while not env.bFinalImageWritten:
                while env.queueImagesWritten.qsize() != env.idx_tick:
                    # pathImage = env.queueImagesWritten.get()
                    # while not os.path.exists(pathImage):
                    #     # print(f'waiting for {pathImage} to exist')
                    #     time.sleep(0.1)
                    # print(f'env.queueImagesWritten.qsize(): {env.queueImagesWritten.qsize()}\tenv.idx_tick: {env.idx_tick}')
                    time.sleep(0.1)
                for actor in env.actor_list:
                    actor.destroy()

                # Append episode reward to a list and log stats (every given number of episodes)
                ep_rewards.append(episode_reward)
                if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                    # Save model, but only when min reward is greater or equal a set value
                    if min_reward >= MIN_REWARD:
                        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


                # Decay epsilon
                if epsilon > MIN_EPSILON:
                    # epsilon *= EPSILON_DECAY
                    epsilon = epsilon_base*(EPSILON_DECAY**episode)
                    epsilon = max(MIN_EPSILON, epsilon)
                if episode == EPISODES:
                    bTrainingComplete = True
                
                print(f'Finished episode {episode} of {EPISODES}')
                print(f'episode: {episode}\treward: {episode_reward}\tframes: {count_frames_completed}\treward/frames: {episode_reward/count_frames_completed}')
                # # fill agent.replay_memory
                # idx_replay_memory = 0
                # while len(agent.replay_memory) < REPLAY_MEMORY_SIZE:
                #     element = agent.replay_memory[idx_replay_memory]
                #     agent.replay_memory.append(element)
                #     idx_replay_memory += 1

            with open(f'tmp/{env.episode:04}.replay_memory', 'wb') as file:
                pickle.dump(agent.replay_memory, file)
            epochs = None
            print(f'len(agent.replay_memory): {len(agent.replay_memory)}')
            if len(agent.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                epochs = 0
            # if idx_action1 < action_size:
            #     epochs = 0
            # else:
            # elif len(agent.replay_memory) >= max(COUNT_FRAME_WINDOW, MINIBATCH_SIZE) and len(agent.replay_memory) < REPLAY_MEMORY_SIZE:
            elif len(agent.replay_memory) >= MINIBATCH_SIZE and len(agent.replay_memory) < REPLAY_MEMORY_SIZE:
                epochs = 100
            if len(agent.replay_memory) == REPLAY_MEMORY_SIZE:
                epochs = 1000
            if epochs > 0:
                # count_batches_completed = previousEpisode_countBatchesTrained
                # print(f'Count of epochs trained: {agent.count_epochs_trained}\tGoal: {agent.count_epochs_trained+epochs}')
                # count_batches_goal = previousEpisode_countBatchesTrained+epochs*REPLAY_MEMORY_SIZE//MINIBATCH_SIZE
                # print(f'Count of batches trained: {agent.count_batches_trained}\tGoal: {count_batches_goal}')
                # for epoch in tqdm(range(1, epochs+1), ascii=True, unit="epoch (parent)"):
                #     loss = []
                #     accuracy = []
                #     thresholdAccuracy = 0.999
                #     thresholdLoss = 1e-5
                #     strMessage = ''
                #     # count_batches_subgoal = count_batches_completed+REPLAY_MEMORY_SIZE//MINIBATCH_SIZE
                #     # for batch in tqdm(range(agent.count_batches_trained, count_batches_subgoal), ascii=True, unit="batch"):
                #     #     agent.train()
                #     #     # if bGAIVI:
                #     #     #     print('\n')
                #     #     #     nvidia_smi = subprocess.Popen('nvidia-smi', shell=True, preexec_fn=os.setsid)
                #     #     count_batches_completed += 1
                #     # while count_batches_completed < count_batches_subgoal:
                #     #     history = agent.train()
                #     #     loss.append(history['loss'][0])
                #     #     accuracy.append(history['accuracy'][0])
                #     #     count_batches_completed += 1
                #     #     # if loss < thresholdLoss:
                #     #     # if statistics.mean(accuracy) > thresholdAccuracy:
                #     #     #     # strMessage += f'Early stop at loss {loss}: {count_batches_completed} of {count_batches_subgoal} batches; '
                #     #     #     strMessage += f'Early stop at accuracy {accuracy}: {count_batches_completed} of {count_batches_subgoal} batches; '
                #     #     #     break
                #     history = agent.train()
                #     loss.append(history['loss'][0])
                #     accuracy.append(history['accuracy'][0])
                #     agent.count_epochs_trained += 1
                #     # if loss < thresholdLoss:
                #     if statistics.mean(accuracy) > thresholdAccuracy:
                #         strMessage += f'Early stop at accuracy {statistics.mean(accuracy)} at {epoch} of {epochs} epochs.'
                #         print(f'{strMessage}\n')
                #         break
                agent.train(epochs=epochs)
                agent.replay_memory.clear()
                    
            previousEpisode_countBatchesTrained = agent.count_batches_trained
            
            # agent.saved_model.save(f'tmp/{env.episode:04}.{agent.count_batches_trained}.model')
            agent.saved_model.save(f'tmp/{env.episode:04}.{agent.count_epochs_trained}.model')
            # print(f'Saved model from episode {env.episode}. Count of batches trained: {agent.count_batches_trained}')
            # print(f'Saved model from episode {env.episode}. Count of epochs trained: {agent.count_epochs_trained}')

    except Exception as e:
        # time.sleep(5)
        print(f'Error message: {e}')
        import traceback
        traceback.print_exc() 
        print(f'Error during episode {env.episode}')
    finally:
        agent.terminate = True
        # trainer_thread.join()
        if bTrainingComplete:
            agent.model.save(f'models/final.model')
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
