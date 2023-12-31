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
from threading import Thread
from tensorflow.keras.callbacks import TensorBoard # Neil modified `from keras.callbacks import TensorBoard`

from tqdm import tqdm


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SHOW_PREVIEW = False
IM_WIDTH = 600
IM_HEIGHT = 600
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
# MODEL_NAME = "Xception"
MODEL_NAME = "Neil_SDC_2023"

MEMORY_FRACTION = 0.8
MIN_REWARD = -200

EPISODES = 100
# EPISODES = 5

DISCOUNT = 0.99
epsilon = 1.0
EPSILON_DECAY = 0.95
# EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

directory = '_out_14rl_custom'
# if os.path.exists(directory):
#     [os.remove(os.path.join(directory, file)) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
# else:
#     print("Directory does not exist or is already removed.")
bSync = False
bVerbose = True


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir) # Neil modified `self.writer = tf.summary.FileWriter(self.log_dir)`

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        _ = [tf.summary.scalar(key, value, step=self.step) for key, value in stats.items()] # Neil commented `self._write_logs(stats, self.step)`


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    episode = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        # self.client.set_timeout(2.0)
        self.client.set_timeout(60)
        # self.client.set_timeout(600)
        # self.world = self.client.get_world()
        self.world = self.client.load_world('Town04_Opt')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        if bSync:
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

        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.transform = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        if bSync:
            self.world.tick()

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_height}")
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

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

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
        i4.save('%s/%03d_%06d.png' % (directory, self.episode, image.frame))

    def step(self, action):
        if bVerbose and False:
            print(f'action: {action}')
            print(f'action[0]: {action[0]}')
            print(f'type(action[0]): {type(action[0])}')
        # if action == 0:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        # elif action == 1:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        # elif action == 2:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
        # elif action == 3:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        # elif action == 4:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=True))
        # self.vehicle.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2]))
        throttle, steer, brake = action
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))

        # if bSync:
        #     self.world.tick() # Neil added

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len (self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        # Neil commented `self.graph = tf.get_default_graph()`

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        # base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Flatten
        base_model = tf.keras.Sequential()
        # base_model.add(Conv2D(4, (3,3), padding='same', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
        base_model.add(Conv2D(1, (3,3), padding='same', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
        base_model.add(BatchNormalization())
        base_model.add(Activation('relu'))
        base_model.add(Flatten())

        x = base_model.output
        # x = GlobalAveragePooling2D()(x)

        throttle = Dense(1, activation="linear", name="throttle")(x)
        steering = Dense(1, activation="linear", name="steering")(x)
        brake = Dense(1, activation="linear", name="brake")(x)

        # predictions = Dense(3, activation="linear")(x)
        # predictions = Dense(5, activation="linear")(x)
        predictions = Concatenate(name="predictions")([throttle, steering, brake])
        model = Model(inputs = base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]) # Neil modified `model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])`
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        # Neil commented `with self.graph.as_default():`
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE) # Neil left tabbed 1

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        # Neil commented `with self.graph.as_default():`
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE) # Neil left tabbed 1

        X = []
        # y = []
        throttle = []
        steer = []
        brake = []

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
            # current_qs[action] = new_q
            current_qs[0] = throttle[index]
            current_qs[1] = steer[index]
            current_qs[2] = brake[index]

            X.append(current_state)
            # y.append(current_qs)
            throttle.append(current_qs[0])
            steer.append(current_qs[1])
            brake.append(current_qs[2])

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # Neil commented `with self.graph.as_default():`
        # self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None) # Neil left tabbed 1
        self.model.fit(
            np.array(X) / 255,
            {"throttle": np.array(throttle), "steering": np.array(steering), "brake": np.array(brake)},
            batch_size=TRAINING_BATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if log_this_step else None
        )


        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        # y = np.random.uniform(size=(1, 5)).astype(np.float32)
        # Neil commented `with self.graph.as_default():`
        self.model.fit(X,y, verbose=False, batch_size=1) # Neil left tabbed 1

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            if not bSync:
                time.sleep(0.01)

if __name__ == "__main__":
    FPS = 60
    ep_rewards = [-200]

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1) # Neil modified `tf.set_random_seed(1)`

    # Neil commented out for CPU `tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)` # Neil modified `gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)`
    pass # Neil modified `backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))`

    if not os.path.isdir("models"):
        os.makedirs("models")

    agent = DQNAgent()
    idx_episode_start = 1
    # if tmp/{episode}.model exists, load model
    import glob
    matching_files = glob.glob(os.path.join('tmp/', '*.model'))
    if len(matching_files) > 0:
        print(f'Load model {matching_files[-1]}')
        agent.model = tf.keras.models.load_model(matching_files[-1])
        idx_episode_start = int(matching_files[-1].split('/')[1].split('.')[0]) + 1

    env = CarEnv()
    # if bSync:
    #     env.world.tick()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))
    bTrainingComplete = False
    try:
        for episode in tqdm(range(idx_episode_start, EPISODES+1), ascii=True, unit="episodes"):
            print(f'Started episode {episode} of {EPISODES}')
            # import subprocess
            # process = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh', shell=True)
            # time.sleep(10)

            # env = CarEnv()
            # agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

            env.collision_hist = []
            agent.tensorboard.step = episode
            env.episode = episode
            episode_reward = 0
            step = 1
            current_state = env.reset()
            # while bSync:
            #     print('tick 1');env.world.tick()
            done = False
            episode_start = time.time()

            while True:
                if bSync:
                    env.world.tick();
                if np.random.random() > epsilon:
                    # action = np.argmax(agent.get_qs(current_state))
                    action = agent.get_qs(current_state)
                else:
                    # action = np.random.randint(0, 3)
                    throttle = np.random.uniform(low=0.0, high=1.0)  # Random throttle value between 0 and 1
                    steer = np.random.uniform(low=-1.0, high=1.0)  # Random steering value between -1 and 1
                    # brake = np.random.uniform(low=0.0, high=1.0)  # Random brake value between 0 and 1
                    brake = np.random.uniform(low=0.0, high=0.0)
                    # action = np.array([[throttle], [steer], [brake]])
                    action = np.array([throttle, steer, brake])
                    if not bSync:
                        time.sleep(1/FPS)

                new_state, reward, done, _ = env.step(action)            
                episode_reward += reward
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                step += 1

                if done:
                    break

            for actor in env.actor_list:
                actor.destroy()
            # process.terminate()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            print(f'Finished episode {episode} of {EPISODES}')
            if episode == EPISODES:
                bTrainingComplete = True

    except Exception as e:
        print(f'Error message: {e}')
        # save episode
        print(f'Error during episode {env.episode}')
        agent.model.save(f'tmp/{env.episode-1}.model')
    
    agent.terminate = True
    trainer_thread.join()
    # agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    if bTrainingComplete:
        agent.model.save(f'models/final.model')