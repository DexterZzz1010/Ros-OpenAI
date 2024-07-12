# Using OpenAI with ROS

# Unit 2: Understanding the ROS + OpenAI structure

In this unit, you are going to follow, step by step, the full workflow of a CartPole simulated environment, including all the environments and scripts involved in its training.

The *openai_ros* package provides a ROS based training structure in order to organize the training of your robots in simulations.

In general, this structure can be divided into two big parts:

- **Training Script**: The training script defines and sets up the Reinforcement Learning (RL) algorithm that you are going to use to train your agent.
- **Training Environment**: The training environment is the ones in charge of providing all the needed data to your learning algorithm from the simulation and the RL Agent environment, in order to make the agent learn. There are different types of Training Environments:
  - **Gazebo Environment**
  - **Robot Environment**
  - **Task Environment**

## Clarification

In order to not confuse you along the course, we need to clarify the following:

- **OpenAI Gym**, is the training environment created by OpenAI to train *systems* with reinforcement learning. It is **not prepared to work with ROS nor Gazebo.**
- The **openai_ros** package, is a ROS package **created by The Construct**, which provides the integration of *OpenAI Gym* with ROS and Gazebo in a simple way.
- The **OpenAI Baselines** are a set of already made RL algorithms, created by OpenAI, which are ready to be used with OpenAI Gym

**In this course, we are going to see how to use the \*openai_ros\* package so you can train ROS based robots with the OpenAI Baselines.**

## Training Script

### Demo 2.1

a) Launch the training script and see how the CartPole starts training.

**Execute in WebShell #1**

```
roslaunch cartpole_v0_training start_training.launch
```

This is the code you launched:

```
<launch>
    <rosparam command="load" file="$(find cartpole_v0_training)/config/cartpole_qlearn_params.yaml" />
    <!-- Launch the training system -->
    <node pkg="cartpole_v0_training" name="cartpole_gym" type="start_training.py" output="screen"/>
</launch>
```

So basically, in the previous demo, you:

1. Loaded the training parameters from the **cartpole_qlearn_params.yaml** file.
2. Launched a Python script called **start_training.py**, which activated the whole learning process.

But... what's the whole structure that holds this training? Let's have a look.

### start_training.py

First of all, let's talk about this **start_training.py** script. This script is the one that **sets up the algorithm that you are going to use in order to make your robot learn**. In this case, we are going to use the [Q-learning algorithm](https://en.wikipedia.org/wiki/Q-learning).

Q-learning is a reinforcement learning technique used in machine learning. The goal of Q-Learning is to learn a **policy**, which tells an **agent** (in our case, a robot) which **action** to take under which circumstances (check the link above for more details).

Next, you can see an example of a training script that uses Qlearn. Try to read the code and understand what it does, by reading the included comments. Then, we'll have a detailed look at the most important parts of it.

```python
#!/usr/bin/env python

import gym
import time
import numpy
import random
import time
import qlearn
from gym import wrappers

# ROS packages required
import rospy
import rospkg
from functools import reduce
# import our training environment
from openai_ros.task_envs.cartpole_stay_up import stay_up


if __name__ == '__main__':
    
    rospy.init_node('cartpole_gym', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('CartPoleStayUp-v0')
    rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('cartpole_v0_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/cartpole_v0/alpha")
    Epsilon = rospy.get_param("/cartpole_v0/epsilon")
    Gamma = rospy.get_param("/cartpole_v0/gamma")
    epsilon_discount = rospy.get_param("/cartpole_v0/epsilon_discount")
    nepisodes = rospy.get_param("/cartpole_v0/nepisodes")
    nsteps = rospy.get_param("/cartpole_v0/nsteps")
    running_step = rospy.get_param("/cartpole_v0/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE => " + str(x))
        
        cumulated_reward = 0  
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        
        # Initialize the environment and get first state of the robot
        
        observation = env.reset()
        state = ''.join(map(str, observation))
        
        # Show on screen the actual situation of the robot
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            
            rospy.loginfo("############### Start Step => "+str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.loginfo ("Next action is: %d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            rospy.loginfo(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            #rospy.logwarn("############### State we were => " + str(state))
            #rospy.logwarn("############### Action that we took => " + str(action))
            #rospy.logwarn("############### Reward that action gave => " + str(reward))
            #rospy.logwarn("############### State in which we will start next step => " + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
                rospy.loginfo ("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.loginfo("############### END Step =>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            #rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logwarn ( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

    
    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
```

### Code Analysis

```python
#!/usr/bin/env python

# 导入所需的模块
import gym
import time
import numpy
import random
import time
from gym import wrappers

# 导入ROS和其他必要的库，用于执行ROS操作
import rospy
import rospkg
from functools import reduce

# 从openai_ros包导入CartPole任务环境

# The Task Environment is the class that defines the task that the robot has to solve (in this case, the task is to maintain the pole upright). 
# We'll have a closer look later at this Task Environment that we are importing here.

from openai_ros.task_envs.cartpole_stay_up import stay_up

# 程序的主入口
if __name__ == '__main__':
    # 初始化一个ROS节点，用于强化学习训练
    rospy.init_node('cartpole_gym', anonymous=True, log_level=rospy.WARN)

    # 创建从任务环境导入的Gym环境
    env = gym.make('CartPoleStayUp-v0')
    rospy.loginfo("Gym环境已创建")

    # 设置用于保存训练结果的目录，使用ROS包路径
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('cartpole_v0_training')
    outdir = pkg_path + '/training_results'

    # 设置环境以记录训练结果
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("监视器Wrapper已启动")

    # 初始化数组，用于跟踪每个周期的最后时间步
    last_time_steps = numpy.ndarray(0)

    # 从ROS参数服务器检索参数，这些参数由ROS启动文件在运行时加载
    Alpha = rospy.get_param("/cartpole_v0/alpha")
    Epsilon = rospy.get_param("/cartpole_v0/epsilon")
    Gamma = rospy.get_param("/cartpole_v0/gamma")
    epsilon_discount = rospy.get_param("/cartpole_v0/epsilon_discount")
    nepisodes = rospy.get_param("/cartpole_v0/nepisodes")
    nsteps = rospy.get_param("/cartpole_v0/nsteps")
    running_step = rospy.get_param("/cartpole_v0/running_step")

    # 初始化Q学习算法，并配置从参数中检索的配置
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    # 跟踪开始时间并初始化最高奖励
    start_time = time.time()
    highest_reward = 0

    # 开始训练，跨越多个周期
    for x in range(nepisodes):
        rospy.logdebug("开始周期 " + str(x))

        # 初始化周期的累积奖励和完成标志
        cumulated_reward = 0
        done = False
        # 应用epsilon折扣，用于探索和利用的权衡
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # 重置环境以开始新周期，并获取初始观测
        observation = env.reset()
        state = ''.join(map(str, observation))

        # 迭代周期的步骤
        for i in range(nsteps):
            rospy.loginfo("开始步骤 " + str(i))
            
            # 根据当前状态选择一个动作
            action = qlearn.chooseAction(state)
            rospy.loginfo("下一个动作是: %d", action)

            # 执行动作并获取新状态、奖励、完成状态和附加信息
            observation, reward, done, info = env.step(action)
            rospy.loginfo(str(observation) + " " + str(reward))

            # 更新累积奖励并检查是否为最高奖励
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            # 准备下一个状态
            nextState = ''.join(map(str, observation))

            # 根据所采取的动作更新Q值
            qlearn.learn(state, action, reward, nextState)

            # 检查周期是否应该结束
            if not done:
                state = nextState
            else:
                rospy.loginfo("周期完成")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.loginfo("结束步骤 " + str(i))

        # 计算并记录周期持续时间
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logwarn("周期: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - 奖励: "+str(cumulated_reward)+"     时间: %d:%02d:%02d" % (h, m, s))

    # 记录所有周期结束后的最终结果
    rospy.loginfo("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| 图片 |")
    l = last_time_steps.tolist()
    l.sort()
    rospy.loginfo("总体得分: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("最佳100得分: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # 关闭环境
    env.close()

```

The **MOST IMPORTANT THING** you need to understand from the training script is that it is totally independent from the environments. The purpose of this training script is to set up the learning algorithm that you want to use in order to make your agent learn, regardless of what is being done in the environments. This means that **you can change the algorithm you use to learn in the training script without having to worry about modifying your environment's struture**. And this is very powerful! Let's test it with the following exercises.

### **Exercise 2.1**

a) In your workspace, create a new package called **my_cartpole_training**.

Execute in WebShell #1

```
catkin_create_pkg my_cartpole_training rospy openai_ros
```

b) Inside the package, create two new folders called **launch** and **config**.

c) Inside the launch folder, create a new file named **start_training.launch**. You can copy the following contents into it:

**start_training.launch**

```
<launch>
    <rosparam command="load" file="$(find my_cartpole_training)/config/cartpole_qlearn_params.yaml" />
    <!-- Launch the training system -->
    <node pkg="my_cartpole_training" name="cartpole_gym" type="start_training.py" output="screen"/> -->
</launch>
```

**start_training.launch**

As you can see, the launch file is quite self-explanatory. What you are doing is, first, loading some parameters that Qlearn needs in order to train, and second, launching the training script that you've reviewed in the above section.

d) The next step will be to create this parameters file for Qlearn. Inside the **config** folder, create a new file named **cartpole_qlearn_params.yaml**. You can copy the following contents into it:

**cartpole_qlearn_params.yaml**

```yaml
cartpole_v0: #namespace

    n_actions: 4
    control_type: "velocity"
    
    #environment variables
    min_pole_angle: -0.7 #-23°
    max_pole_angle: 0.7 #23°
    
    max_base_velocity: 50
    max_base_pose_x: 2.5
    min_base_pose_x: -2.5
    
    # those parameters are very important. They are affecting the learning experience
    # They indicate how fast the control can be
    # If the running step is too large, then there will be a long time between 2 ctrl commans
    # If the pos_step is too large, then the changes in position will be very abrupt
    running_step: 0.04 # amount of time the control will be executed
    pos_step: 0.016     # increment in position for each command
    
    #qlearn parameters
    alpha: 0.5
    gamma: 0.9
    epsilon: 0.1
    epsilon_discount: 0.999
    nepisodes: 1000
    nsteps: 1000
    number_splits: 10 #set to change the number of state splits for the continuous problem and also the number of env_variable splits

    init_pos: 0.0 # Position in which the base will start
    wait_time: 0.1 # Time to wait in the reset phases
```

**cartpole_qlearn_params.yaml**

In this file, we are just setting some parameters that we'll need for our training. These parameters will first be loaded into the ROS parameter server, and later retrieved by our Training Environments.

e) Now, you will need to place a file that defines the Qlearn algorithm inside the **src** folder of your package. Create a file named **qlearn.py** here, and place the following contents into it:

**qlearn.py**

```python
'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
 
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
 
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import random

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
```

This file basically defines the Qlearn algorithm. It is imported into our training script in the following line:



```
import qlearn
```

Q-Learn is a very basic RL algorithm. If you want to learn more about how Q-Learn works, check [this link](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56).

```python
import random

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}  # 初始化一个空字典来存储状态-动作对的Q值
        self.epsilon = epsilon  # 探索常数，控制学习过程中探索和利用的平衡
        self.alpha = alpha      # 学习率，控制Q值更新的步长
        self.gamma = gamma      # 折现因子，衡量未来奖励的当前价值
        self.actions = actions  # 可能的动作集

    def getQ(self, state, action):
        # 从Q表中获取给定状态-动作对的Q值，如果不存在则返回0.0
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        使用以下Q学习公式更新Q值：
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s')) - Q(s, a))
        其中:
        - reward(s,a) 是从环境获得的即时奖励
        - max(Q(s')) 是下一个状态的最大Q值
        - Q(s, a) 是当前状态-动作对的Q值
        '''
        oldv = self.q.get((state, action), None)  # 获取当前的Q值
        if oldv is None:
            self.q[(state, action)] = reward  # 如果当前状态-动作对没有Q值，初始化为获得的奖励
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)  # 更新Q值

    def chooseAction(self, state, return_q=False):
        # 为给定状态计算每个动作的Q值
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)  # 找到最大的Q值

        if random.random() < self.epsilon:
            # 以epsilon的概率进行探索，添加随机性以避免局部最优
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # 如果有多个动作具有相同的最大Q值，则随机选择一个
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:  # 如果需要返回Q值
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        # 从所有可能的动作中找到在下一个状态下的最大Q值
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        # 使用learnQ方法更新Q值
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

```



f) Finally, also inside the **src** folder of your package, you'll have to place the **start_training.py** script that we've been reviewing. You will need to modify the name of the package where to store the training results:

```
pkg_path = rospack.get_path('my_cartpole_training')
```

g) Also create the **training_results** folder inside your package.

Execute in WebShell #1

```
roscd my_cartpole_training
```

```
mkdir training_results
```

f) And that's it! Now you have created a package in order to reproduce the demo you executed in the first step. Run now your launch file and see how the CartPole starts to learn.

### **Exercise 2.2**

a) Now, let's change the learning algorithm that we use for learning. In this case, instead of the Qlearn algorithm, we'll use the sarsa algorithm. Below, you have the file that defines the sarsa learning algorithm. Create a new file named *sarsa.py* and include its content.

b)Make the necessary changes to your package in order to use the sarsa algorithm for training, instead of the Q-learn one.

**sarsa.py**

```python
import random


class Sarsa:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward 
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.getQ(state2, action2)
        self.learnQ(state1, action1, reward, reward + self.gamma * qnext)
```



import random

class Sarsa:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        # 初始化SARSA学习算法的参数
        self.q = {}  # Q值字典，用来存储状态-动作对的Q值

```python
    # 探索率epsilon：控制学习过程中的探索与利用的平衡
    self.epsilon = epsilon
    # 学习率alpha：影响Q值更新的幅度
    self.alpha = alpha
    # 折扣因子gamma：决定未来奖励的现值，表示未来奖励的重要性
    self.gamma = gamma
    # 可执行的动作列表
    self.actions = actions

def getQ(self, state, action):
    # 获取指定状态-动作对的Q值，如果不存在则返回0.0
    return self.q.get((state, action), 0.0)

def learnQ(self, state, action, reward, value):
    # 根据当前的经验更新Q值
    oldv = self.q.get((state, action), None)  # 获取旧的Q值
    if oldv is None:
        # 如果Q值不存在，则直接设置为当前获得的奖励
        self.q[(state, action)] = reward
    else:
        # 否则，根据学习率alpha和计算得到的TD目标（value）更新Q值
        self.q[(state, action)] = oldv + self.alpha * (value - oldv)

def chooseAction(self, state):
    # 根据当前状态选择一个动作
    if random.random() < self.epsilon:
        # 以epsilon的概率进行随机选择，实现探索
        action = random.choice(self.actions)
    else:
        # 否则，选择具有最高Q值的动作，实现利用
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        count = q.count(maxQ)
        if count > 1:
            # 如果有多个动作具有相同的最高Q值，随机选择一个
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
    return action

def learn(self, state1, action1, reward, state2, action2):
    # 执行学习过程，更新Q值
    qnext = self.getQ(state2, action2)  # 获取下一个状态-动作对的Q值
    # 调用learnQ方法更新当前状态-动作对的Q值，使用当前奖励和折扣后的下一个Q值
    self.learnQ(state1, action1, reward, reward + self.gamma * qnext)
```

c) Launch your Training script again and check if you see any differences between the performances of Qlearn and sarsa.



## 训练环境

在之前的代码中，你看到我们启动了一个名为 **CartPoleStayUp-v0** 的环境。但这个环境是用来做什么的？它有什么作用呢？

简而言之，一个环境就是一个问题（比如让CartPole保持直立）以及一个最小化的界面，智能体（如机器人）可以与之交互。OpenAI Gym 中的环境被设计用来客观地测试和评估智能体的能力。

在OpenAI Gym中，通常你可以只用一个环境来定义一切。但我们的情况更为复杂，因为我们希望将OpenAI与ROS和Gazebo整合。

这就是为什么我们创建了一个结构来组织并分割OpenAI Gym环境为三种不同类型：

- 任务环境：

  这个环境致力于实现特定训练会话将使用的所有功能。例如，使用相同的机器人环境，你可以创建不同的任务环境，以便用不同的方法（动作）训练你的机器人完成不同的任务。

  - *你需要为不同的任务拥有不同的任务环境*

- 机器人环境：

  这个环境致力于实现特定机器人在训练期间将使用的所有功能。但是，它不包括那些依赖于你将要实施的训练的具体任务。

  - *你需要为不同的机器人拥有不同的机器人环境*

- **Gazebo环境**：这个环境适用于你想要实施的任何训练或机器人。这个环境将生成你的机器人与Gazebo之间的所有连接，这样你就不必担心这方面的问题

![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/pipeline.png)



It's very important that you understand that this environment structure will always be the same, regardless of:

- the problem you want to solve
- the robot you want to train
- the learning algorithm that you want to use

Also, these three environments will be connected to each other in the sense that each environment will **inherit** from the previous one. It goes like this:

- **Task Environment** inherits from the **Robot Environment**.

- **Robot Environment** inherits from the **Gazebo Environment**.

- **Gazebo Environment** inherits from the **Gym Environment**. The Gym Environment (gym.Env) is the most basic Environment structure provided by OpenAI Gym framework.

























































