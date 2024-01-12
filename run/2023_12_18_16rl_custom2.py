import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
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

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

bSAMBHU24 = True
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
    number_of_lines = len(file.readlines())
# MIN_REPLAY_MEMORY_SIZE = int(1.5 * number_of_lines)
# MIN_REPLAY_MEMORY_SIZE = int(64 * number_of_lines)
# REPLAY_MEMORY_SIZE = 5*number_of_lines
REPLAY_MEMORY_SIZE = 50_000
if bSAMBHU24:
    MINIBATCH_SIZE = 128 # 6 GB GPU memory
else:
    # MINIBATCH_SIZE = 128*2*8
    MINIBATCH_SIZE = 20_000
    # MINIBATCH_SIZE = 5_000
    # MINIBATCH_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 20_000
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
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
action_space = {'throttle': np.linspace(0.0, 1.0, num=11),
# action_space = {'throttle': np.linspace(0.0, 1.0, num=2),
                'steer': np.linspace(-1.0, 1.0, num=21),
                # 'steer': np.linspace(-1.0, 1.0, num=3),
                'brake': np.linspace(0.0, 1.0, num=11)}
                # 'brake': np.linspace(0.0, 0.0, num=11)}
                # 'brake': np.linspace(0.0, 1.0, num=2)}
# print(action_space);import sys;sys.exit()
action_size = len(action_space['throttle'])*len(action_space['steer'])*len(action_space['brake'])

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    episode = None
    action_space = action_space
    idx_tick = -1

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
        i4.save('%s/%04d/%06d.png' % (directory_output, self.episode, image.frame))
        while not os.path.exists('%s/%04d/%06d.png' % (directory_output, self.episode, image.frame)):
            time.sleep(0.1)

    def step(self, action):
        if bVerbose and False:
            print(f'action: {action}')
            print(f'action[0]: {action[0]}')
            print(f'type(action[0]): {type(action[0])}')
        throttle_action = action // (len(action_space['steer'])*len(action_space['brake']))
        steer_action = (action % (len(action_space['steer'])*len(action_space['brake']))) // len(action_space['brake'])
        brake_action = action % len(action_space['brake'])

        throttle_value = self.action_space['throttle'][throttle_action]
        steer_value = self.action_space['steer'][steer_action]
        brake_value = self.action_space['brake'][brake_action]

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
        # reward = -1*distance**3 - distance + 1
        if distance < 10:
            reward += 10
        else:
            reward -= 10

        # Set 'done' flag to True when ticks exceed the lines in the file
        # done = self.idx_tick >= len(lines)
        done = self.idx_tick >= 1000

        # v = self.vehicle.get_velocity()
        # kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        # reward += kmh
        # if kmh > 15 and kmh < 50:
        #     reward = 15

        # if self.episode_start + SECONDS_PER_EPISODE < time.time():
        # if self.episode_start + SECONDS_PER_EPISODE < self.world.get_snapshot().timestamp.elapsed_seconds:
        #     done = True

        if len (self.collision_hist) != 0:
            done = True
            # reward = -200
            reward = -0.001
            # reward = -1

        return self.front_camera, reward, done, None

if bGPU:
    if not bGAIVI:
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f'GPUs: {gpus}')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
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
            # base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
            from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Flatten, AveragePooling2D, MaxPooling2D
            base_model = tf.keras.Sequential()
            # base_model.add(Conv2D(1, (3,3), padding='same', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
            # base_model.add(AveragePooling2D(pool_size=(4,4), input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
            count_filters = 64
            base_model.add(Conv2D(count_filters, (3,3), padding='same', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
            base_model.add(MaxPooling2D(pool_size=(2, 2)))
            base_model.add(BatchNormalization())
            base_model.add(Activation('relu'))
            
            base_model.add(Conv2D(count_filters, (3,3), padding='same'))
            base_model.add(MaxPooling2D(pool_size=(2, 2)))
            base_model.add(BatchNormalization())
            base_model.add(Activation('relu'))

            base_model.add(Conv2D(count_filters, (3,3), padding='same'))
            base_model.add(MaxPooling2D(pool_size=(2, 2)))
            base_model.add(BatchNormalization())
            base_model.add(Activation('relu'))

            base_model.add(Conv2D(count_filters, (3,3), padding='same'))
            base_model.add(MaxPooling2D(pool_size=(2, 2)))
            base_model.add(BatchNormalization())
            base_model.add(Activation('relu'))

            x = base_model.output
            x = Flatten()(x)

            # print(f'x.shape: {x.shape}')

            size_reduce = 2
            while(x.shape.as_list()[1] >= size_reduce*(action_size+1)):
                x = Dense(x.shape.as_list()[1]//size_reduce, activation="relu")(x)

            predictions = Dense(action_size, activation="linear")(x)
            model = Model(inputs = base_model.input, outputs=predictions)
            model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]) # Neil modified `model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])`
            # print(model.summary())
            return model

        def update_replay_memory(self, transition):
            # transition = (current_state, action, reward, new_state, done)
            self.replay_memory.append(transition)

        def train(self):
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return

            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

            # current_states = np.array([transition[0] for transition in minibatch])/255
            current_states = np.array([transition[0] for transition in minibatch])
            # Neil commented `with self.graph.as_default():`
            # current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE) # Neil left tabbed 1
            # current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE, verbose=0)
            current_qs_list = None
            try:
                current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE, verbose=0)
            except Exception as e:
                print(f'Error message: {e}')
            
            # new_current_states = np.array([transition[3] for transition in minibatch])/255
            new_current_states = np.array([transition[3] for transition in minibatch])
            # Neil commented `with self.graph.as_default():`
            # future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE) # Neil left tabbed 1
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE, verbose=0)

            X = []
            y = []

            for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
                if bVerbose and False:
                    print(f'action: {action}')
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

                X.append(current_state)
                y.append(current_qs)

            log_this_step = False

            # Neil commented `with self.graph.as_default():`
            # self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None) # Neil left tabbed 1
            self.model.fit(
                # np.array(X) / 255,
                np.array(X),
                np.array(y),
                batch_size=TRAINING_BATCH_SIZE,
                verbose=0,
                shuffle=False,
                # callbacks=[self.tensorboard] if log_this_step else None # 12/18/2023 7:47 PM: Neil commented out
            )


            if log_this_step:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

            if self.count_batches_trained == 0:
                print('Finished training first batch.')
            self.count_batches_trained += 1
            time.sleep(0.01)
            self.saved_model.set_weights(self.model.get_weights())
            self.count_saved_models += 1

        def get_qs(self, state):
            # print(state.shape)
            # print(np.array(state).shape)
            # print(type(state))
            # print(state)
            # from PIL import Image
            # im = Image.fromarray(state)
            # im.save('img.png');import sys;sys.exit()
            # return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
            # print(self.model.predict(np.expand_dims(state, axis=0), verbose=0));import sys;sys.exit()
            # return self.model.predict(state, verbose=0)[0]
            return self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

        def train_in_loop(self):
            # X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
            X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.uint8)
            # y = np.random.uniform(size=(1, action_size)).astype(np.float32)
            y = np.random.uniform(size=(1, action_size)).astype(np.ushort)
            # Neil commented `with self.graph.as_default():`
            self.model.fit(X,y, verbose=False, batch_size=1) # Neil left tabbed 1

            self.training_initialized = True

            while True:
                if self.terminate:
                    return
                self.train()
                if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                    time.sleep(1)
                else:
                    if self.count_batches_trained == 0:
                        print('Finished training first epoch.')
                        if bGAIVI:
                            nvidia_smi = subprocess.Popen('nvidia-smi', shell=True)
                            nvidia_smi.wait()
                            time.sleep(1)
                    self.count_batches_trained += 1
                    self.saved_model.set_weights(self.model.get_weights())
                    self.count_saved_models += 1

if __name__ == "__main__":
    FPS = 60
    ep_rewards = [-200]
    # ep_rewards = [-0.0001]

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1) # Neil modified `tf.set_random_seed(1)`

    agent = DQNAgent()
    x = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.uint8)
    y = np.random.uniform(size=(1, action_size)).astype(np.ushort)
    agent.model.fit(x, y, verbose=False, batch_size=1) # Neil left tabbed 1

    idx_episode_start = 1
    idx_action = 0
    import glob, shutil
    bLoadReplayMemory = True
    if bLoadReplayMemory:
        with open('bak/0282.replay_memory', 'rb') as file:
            agent.replay_memory = pickle.load(file)
        idx_episode_start = 283
        idx_action = action_size
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

        matching_files = glob.glob(os.path.join('tmp', '*.idx_action'))
        matching_files.sort()
        print(f'Index action in temp {matching_files}')
        print(f'load idx_action {matching_files[-1]}')
        idx_action = matching_files[-1].split('/')[1].split('.')[0]
        idx_action = int(idx_action)

        idx_episode_start = idx_episode_crashed

    env = CarEnv()

    # trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    # trainer_thread.start()

    # while not agent.training_initialized:
    #     time.sleep(0.01)

    bTrainingComplete = False
    previousEpisode_countBatchesTrained = agent.count_batches_trained
    count_framesPerAction = 0
    max_framesPerAction = 100
    try:
        for episode in tqdm(range(idx_episode_start, EPISODES+1), ascii=True, unit="episode"):
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
                # nvidia_smi.wait()

            env.collision_hist = []
            # agent.tensorboard.step = episode
            env.episode = episode
            episode_reward = 0
            # step = 1
            current_state = env.reset()
            if bSync and False:
                env.world.tick()
                env.idx_tick += 1
            done = False
            idx_control = 0
            # for i in range(0,10):
            #     env.world.tick()

            while True:
                if bSync and False:
                    # print(f'bSync inside episode')
                    env.world.tick();
                    env.idx_tick += 1
                action = None
                # # if np.random.random() > epsilon and True:
                # if len(agent.replay_memory) <= REPLAY_MEMORY_SIZE:
                # # if False:
                #     # action = np.argmax(agent.get_qs(current_state))
                #     with open('_out_07CARLA_AP/Controls_Town04_0_335.txt', 'r') as file:
                #         lines = file.readlines()
                #         throttle,steer,brake = lines[idx_control].split()
                #         idx_control += 1
                #         throttle,steer,brake = float(throttle),float(steer),float(brake)
                #         throttle_index = np.abs(action_space['throttle'] - throttle).argmin()
                #         steer_index = np.abs(action_space['steer'] - steer).argmin()
                #         brake_index = np.abs(action_space['brake'] - brake).argmin()
                #         action = throttle_index * len(action_space['steer']) * len(action_space['brake']) + \
                #                     steer_index * len(action_space['brake']) + \
                #                     brake_index
                # else:
                #     # bActionValid = False
                #     # while not bActionValid:
                #     #     action = np.random.randint(0, action_size)
                #     #     throttle_action = action // (len(action_space['steer'])*len(action_space['brake']))
                #     #     brake_action = action % len(action_space['brake'])
                #     #     if throttle_action == 0 or brake_action == 0:
                #     #         bActionValid = True
                #     action = np.argmax(agent.get_qs(current_state))                    
                #     if not bSync:
                #         time.sleep(1/FPS)
                if idx_action < action_size:
                    count_framesPerAction += 1
                    if count_framesPerAction > max_framesPerAction:
                        print(f'Finished idx_action: {idx_action}\taction size: {action_size}')
                        idx_action += 1
                        count_framesPerAction = 0
                    bActionValid = False
                    while not bActionValid and idx_action < action_size:
                        action = idx_action
                        throttle_action = action // (len(action_space['steer'])*len(action_space['brake']))
                        brake_action = action % len(action_space['brake'])
                        if brake_action == 0 and throttle_action > 0:
                            bActionValid = True
                            matching_files = glob.glob(os.path.join('tmp', '*idx_action'))
                            [os.remove(matching_file) for matching_file in matching_files]
                            open(f'tmp/{idx_action:04d}.idx_action', "w")
                        else:
                            idx_action += 1
                    if idx_action >= action_size:
                        print(f'Finished initializing all actions. Predicting from model.')
                        action = np.argmax(agent.get_qs(current_state))                    
                else:
                    action = np.argmax(agent.get_qs(current_state))                    

                new_state, reward, done, _ = env.step(action)            
                episode_reward += reward
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                # agent.train_in_loop() # 12/19/2023 2:00 AM: Neil added
                # step += 1

                if done:
                    break

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
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            if episode == EPISODES:
                bTrainingComplete = True
            
            print(f'Finished episode {episode} of {EPISODES}')
            
            epochs = None
            print(f'len(agent.replay_memory): {len(agent.replay_memory)}')
            if len(agent.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                epochs = 0
            elif len(agent.replay_memory) < REPLAY_MEMORY_SIZE:
                if bSAMBHU24:
                    epochs = 1
                    # epochs = 0
                else:
                    epochs = int(1e3)
            if len(agent.replay_memory) == REPLAY_MEMORY_SIZE:
                if bSAMBHU24:
                    epochs = 10
                else:
                    epochs = int(1e6)
            if epochs > 0:
                count_batches_completed = previousEpisode_countBatchesTrained
                print(f'Count of epochs trained: {agent.count_epochs_trained}\tGoal: {agent.count_epochs_trained+epochs}')
                count_batches_goal = previousEpisode_countBatchesTrained+epochs*REPLAY_MEMORY_SIZE//MINIBATCH_SIZE
                # print(f'Count of batches trained: {agent.count_batches_trained}\tGoal: {count_batches_goal}')
                for epoch in tqdm(range(1, epochs+1), ascii=True, unit="epoch"):
                    count_batches_subgoal = count_batches_completed+REPLAY_MEMORY_SIZE//MINIBATCH_SIZE
                    # for batch in tqdm(range(agent.count_batches_trained, count_batches_subgoal), ascii=True, unit="batch"):
                    #     agent.train()
                    while count_batches_completed < count_batches_subgoal:
                        agent.train()
                        count_batches_completed += 1
                    agent.count_epochs_trained += 1
                    
            previousEpisode_countBatchesTrained = agent.count_batches_trained
            
            # print(f'agent.count_saved_models: {agent.count_saved_models}')
            # time.sleep(6)
            # print(f'agent.count_saved_models: {agent.count_saved_models}')
            
            # agent.saved_model.save(f'tmp/{env.episode:04}.{agent.count_batches_trained}.model')
            agent.saved_model.save(f'tmp/{env.episode:04}.{agent.count_epochs_trained}.model')
            # print(f'Saved model from episode {env.episode}. Count of batches trained: {agent.count_batches_trained}')
            print(f'Saved model from episode {env.episode}. Count of epochs trained: {agent.count_epochs_trained}')

            with open(f'tmp/{env.episode:04}.replay_memory', 'wb') as file:
                pickle.dump(agent.replay_memory, file)

            # import subprocess
            # git = subprocess.Popen('git commit -a -m \"upload results\"')
            # git.wait()
            # git = subprocess.Popen('git push')
            # git.wait()

    except Exception as e:
        print(f'Error message: {e}')
        # save episode
        print(f'Error during episode {env.episode}')
    finally:
        agent.terminate = True
        # trainer_thread.join()
        if bTrainingComplete:
            agent.model.save(f'models/final.model')
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
