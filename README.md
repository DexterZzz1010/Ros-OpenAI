# Using OpenAI with ROS

# Unit 1: Understanding the ROS + OpenAI structure

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

## Task Environment

The Task Environment is the highest one in the inheritance structure, which also means that it is the most specific one. This environent fully depends on the task we want the robot learn.

Next, you can see an example of a Training Environment that trains the Pole to stay up on the Cart.

**stay_up.py**

```python
from gym import utils
from openai_ros.robot_envs import cartpole_env
from gym.envs.registration import register
from gym import error, spaces
import rospy
import math
import numpy as np

# The path is __init__.py of openai_ros, where we import the CartPoleStayUpEnv directly
register(
        id='CartPoleStayUp-v0',
        entry_point='openai_ros:CartPoleStayUpEnv',
        max_episode_steps=1000,
    )

class CartPoleStayUpEnv(cartpole_env.CartPoleEnv):
    def __init__(self):
        
        self.get_params()
        
        self.action_space = spaces.Discrete(self.n_actions)
        
        cartpole_env.CartPoleEnv.__init__(
            self, control_type=self.control_type
            )
            
    def get_params(self):
        #get configuration parameters
        self.n_actions = rospy.get_param('/cartpole_v0/n_actions')
        self.min_pole_angle = rospy.get_param('/cartpole_v0/min_pole_angle')
        self.max_pole_angle = rospy.get_param('/cartpole_v0/max_pole_angle')
        self.max_base_velocity = rospy.get_param('/cartpole_v0/max_base_velocity')
        self.min_base_pose_x = rospy.get_param('/cartpole_v0/min_base_pose_x')
        self.max_base_pose_x = rospy.get_param('/cartpole_v0/max_base_pose_x')
        self.pos_step = rospy.get_param('/cartpole_v0/pos_step')
        self.running_step = rospy.get_param('/cartpole_v0/running_step')
        self.init_pos = rospy.get_param('/cartpole_v0/init_pos')
        self.wait_time = rospy.get_param('/cartpole_v0/wait_time')
        self.control_type = rospy.get_param('/cartpole_v0/control_type')
        
    def _set_action(self, action):
        
        # Take action
        if action == 0: #LEFT
            rospy.loginfo("GO LEFT...")
            self.pos[0] -= self.pos_step
        elif action == 1: #RIGHT
            rospy.loginfo("GO RIGHT...")
            self.pos[0] += self.pos_step
        elif action == 2: #LEFT BIG
            rospy.loginfo("GO LEFT BIG...")
            self.pos[0] -= self.pos_step * 10
        elif action == 3: #RIGHT BIG
            rospy.loginfo("GO RIGHT BIG...")
            self.pos[0] += self.pos_step * 10
            
            
        # Apply action to simulation.
        rospy.loginfo("MOVING TO POS=="+str(self.pos))

        # 1st: unpause simulation
        rospy.logdebug("Unpause SIM...")
        self.gazebo.unpauseSim()

        self.move_joints(self.pos)
        rospy.logdebug("Wait for some time to execute movement, time="+str(self.running_step))
        rospy.sleep(self.running_step) #wait for some time
        rospy.logdebug("DONE Wait for some time to execute movement, time=" + str(self.running_step))

        # 3rd: pause simulation
        rospy.logdebug("Pause SIM...")
        self.gazebo.pauseSim()

    def _get_obs(self):
        
        data = self.joints
        #       base_postion                base_velocity              pole angle                 pole velocity
        obs = [round(data.position[1],1), round(data.velocity[1],1), round(data.position[0],1), round(data.velocity[0],1)]

        return obs
        
    def _is_done(self, observations):
        done = False
        data = self.joints
        
        rospy.loginfo("BASEPOSITION=="+str(observations[0]))
        rospy.loginfo("POLE ANGLE==" + str(observations[2]))
        if (self.min_base_pose_x >= observations[0] or observations[0] >= self.max_base_pose_x): #check if the base is still within the ranges of (-2, 2)
            rospy.logerr("Base Outside Limits==>min="+str(self.min_base_pose_x)+",pos="+str(observations[0])+",max="+str(self.max_base_pose_x))
            done = True
        if (self.min_pole_angle >= observations[2] or observations[2] >= self.max_pole_angle): #check if pole has toppled over
            rospy.logerr(
                "Pole Angle Outside Limits==>min=" + str(self.min_pole_angle) + ",pos=" + str(observations[2]) + ",max=" + str(
                    self.max_pole_angle))
            done = True
            
        return done
        
    def _compute_reward(self, observations, done):

        """
        Gives more points for staying upright, gets data from given observations to avoid
        having different data than other previous functions
        :return:reward
        """
        
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        
        return reward
        
    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.steps_beyond_done = None
        
    def _set_init_pose(self):
        """
        Sets joints to initial position [0,0,0]
        :return:
        """
        
        self.check_publishers_connection()
        
        # Reset Internal pos variable
        self.init_internal_vars(self.init_pos)
        self.move_joints(self.pos)
```



代码解释：

```python
from gym import utils
from openai_ros.robot_envs import cartpole_env
from gym.envs.registration import register
from gym import error, spaces
import rospy
import math
import numpy as np

# 注册CartPoleStayUp-v0环境到OpenAI Gym
register(
        id='CartPoleStayUp-v0',  # 环境的唯一标识符
        entry_point='openai_ros:CartPoleStayUpEnv',  # 创建环境实例的入口点
        max_episode_steps=1000,  # 每个episode的最大步数限制
    )

class CartPoleStayUpEnv(cartpole_env.CartPoleEnv):
    def __init__(self):
        """
        初始化CartPoleStayUp环境实例。
        这个构造函数首先从ROS参数服务器获取所有需要的环境参数，
        然后设置动作空间，并调用基类构造函数来完成进一步的初始化。
        """
        self.get_params()  # 获取环境相关参数
        self.action_space = spaces.Discrete(self.n_actions)  # 定义动作空间
        cartpole_env.CartPoleEnv.__init__(self, control_type=self.control_type)  # 调用基类构造函数
            
    def get_params(self):
        """
        从ROS参数服务器获取所有需要的环境参数。
        这些参数包括动作数量、杆的角度限制、基座的速度和位置限制等，
        用于配置和控制CartPole的物理模型行为。
        """
        self.n_actions = rospy.get_param('/cartpole_v0/n_actions')
        self.min_pole_angle = rospy.get_param('/cartpole_v0/min_pole_angle')
        self.max_pole_angle = rospy.get_param('/cartpole_v0/max_pole_angle')
        self.max_base_velocity = rospy.get_param('/cartpole_v0/max_base_velocity')
        self.min_base_pose_x = rospy.get_param('/cartpole_v0/min_base_pose_x')
        self.max_base_pose_x = rospy.get_param('/cartpole_v0/max_base_pose_x')
        self.pos_step = rospy.get_param('/cartpole_v0/pos_step')
        self.running_step = rospy.get_param('/cartpole_v0/running_step')
        self.init_pos = rospy.get_param('/cartpole_v0/init_pos')
        self.wait_time = rospy.get_param('/cartpole_v0/wait_time')
        self.control_type = rospy.get_param('/cartpole_v0/control_type')
        
    def _set_action(self, action):
        """
        执行指定的动作，动作编号由 'action' 参数指定。
        根据动作编号调整CartPole的位置，模拟不同的移动指令。
        """
        # 根据动作编号调整购物车位置
        if action == 0:
            rospy.loginfo("GO LEFT...")
            self.pos[0] -= self.pos_step
        elif action == 1:
            rospy.loginfo("GO RIGHT...")
            self.pos[0] += self.pos_step
        elif action == 2:
            rospy.loginfo("GO LEFT BIG...")
            self.pos[0] -= self.pos_step * 10
        elif action == 3:
            rospy.loginfo("GO RIGHT BIG...")
            self.pos[0] += self.pos_step * 10

        # 更新仿真环境中的位置
        rospy.loginfo("MOVING TO POS==" + str(self.pos))
        rospy.logdebug("Unpause SIM...")
        self.gazebo.unpauseSim()
        self.move_joints(self.pos)
        rospy.logdebug("Wait for some time to execute movement, time=" + str(self.running_step))
        rospy.sleep(self.running_step)
        rospy.logdebug("DONE Wait for some time to execute movement, time=" + str(self.running_step))
        rospy.logdebug("Pause SIM...")
        self.gazebo.pauseSim()

    def _get_obs(self):
        """
        获取并返回当前环境的观测值，包括倒立摆的位置和速度，杆的角度和速度。
        这些数据是从仿真环境中的关节状态获取的。
        """
        data = self.joints
        obs = [round(data.position[1], 1), round(data.velocity[1], 1), round(data.position[0], 1), round(data.velocity[0], 1)]
        return obs
        
    def _is_done(self, observations):
        """
        根据当前观测判断episode是否应该结束。
        如果购物车超出了设定的位置范围或杆的角度超过了限制，则结束episode。
        """
        done = False
        rospy.loginfo("BASEPOSITION==" + str(observations[0]))
        rospy.loginfo("POLE ANGLE==" + str(observations[2]))
        if (self.min_base_pose_x >= observations[0] or observations[0] >= self.max_base_pose_x) or \
           (self.min_pole_angle >= observations[2] or observations[2] >= self.max_pole_angle):
            rospy.logerr("Limit exceeded")
            done = True
        return done
        
    def _compute_reward(self, observations, done):
        """
        根据当前的状态计算并返回奖励。
        如果任务没有完成（即杆未倒），则继续给予奖励。如果完成，则根据完成后的步数调整奖励。
        """
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0
        return reward
        
    def _init_env_variables(self):
        """
        初始化或重置环境变量，为新的episode做准备。
        主要是重置完成步数的计数器。
        """
        self.steps_beyond_done = None
        
    def _set_init_pose(self):
        """
        设置机器人的初始位置。
        确保机器人的起始位置符合任务要求，为仿真的准确执行奠定基础。
        """
        self.check_publishers_connection()  # 确保ROS节点正常连接
        self.init_internal_vars(self.init_pos)  # 初始化内部变量为初始位置
        self.move_joints(self.init_pos)  # 移动关节到初始位置

```







It's very important that you understand that you can add any function that you want into this Task Environment. However, there are some functions that will **ALWAYS** need to be defined here because they are required for the rest of the Environments to work properly. These functions are:

- The **_set_action()** function
- The **_get_obs()** function
- The **_is_done()** function
- The **_compute_reward()** function
- The **_set_init_pose()** function
- The **_init_env_variables()** function

If for your specific task, you don't need to implement some of these functions, you can just make them **pass** (like we do in this case with the _set_init_pose() and _init_env_variables() functions). But you will need to add them to your code anyway, or you will get a **NotImplementedError()**.



### **Exercise 2.3**

a) Inside **src** folder of the **my_cartpole_training** package, add a new script called **my_cartpole_task_env.py**. Into this new file, copy the contents of the [**stay_up.py**](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/Course_OpenAIBaselines_Unit1.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=ab4CkxYwJI0jSUIOOeO20fxV9KI%3D&Expires=1720800981#stay-up) script that you have just reviewed.

b) Now, modify the Task Environment code so that it takes only two actions: move right and move left.

**NOTE**: Also, you will need to modify the registration id of the environment, in order to avoid conflicts. You just need to change the **v0** for a **v1**.



## Robot Environment

The robot environment will then contain all the functions associated with the specific "robot" that you want to train. This means that it will contain all the functionalities that your robot will need in order to be controlled.

For instance, let's focus on the CartPole example. Let's say that in order to be able to control a CartPole environment, we will basically need three things:

- A way to check that all the systems (ROS topics, Publishers, etc.) are ready and working okay
- A way to move the Cart alongside the rails
- A way to get all the data about all the sensors that we want to evaluate (including the position of the Pole)

For this case, then, we would need to define all these functions into the Robot Environment. Let's see an example of a Robot Environment script that takes into account the three points introduced above. You can check out the code here:



**cartpole_env.py**

```python
#!/usr/bin/env python

import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import utils, spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gym.utils import seeding
from gym.envs.registration import register
import copy
import math
import os

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState
from rosgraph_msgs.msg import Clock
from openai_ros import robot_gazebo_env

class CartPoleEnv(robot_gazebo_env.RobotGazeboEnv):
    def __init__(
        self, control_type
    ):
        
        self.publishers_array = []
        self._base_pub = rospy.Publisher('/cartpole_v0/foot_joint_velocity_controller/command', Float64, queue_size=1)
        self._pole_pub = rospy.Publisher('/cartpole_v0/pole_joint_velocity_controller/command', Float64, queue_size=1)
        self.publishers_array.append(self._base_pub)
        self.publishers_array.append(self._pole_pub)
        
        rospy.Subscriber("/cartpole_v0/joint_states", JointState, self.joints_callback)
        
        self.control_type = control_type
        if self.control_type == "velocity":
            self.controllers_list = ['joint_state_controller',
                                    'pole_joint_velocity_controller',
                                    'foot_joint_velocity_controller',
                                    ]
                                    
        elif self.control_type == "position":
            self.controllers_list = ['joint_state_controller',
                                    'pole_joint_position_controller',
                                    'foot_joint_position_controller',
                                    ]
                                    
        elif self.control_type == "effort":
            self.controllers_list = ['joint_state_controller',
                                    'pole_joint_effort_controller',
                                    'foot_joint_effort_controller',
                                    ]

        self.robot_name_space = "cartpole_v0"
        self.reset_controls = True

        # Seed the environment
        self._seed()
        self.steps_beyond_done = None
        
        super(CartPoleEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=self.reset_controls
            )

    def joints_callback(self, data):
        self.joints = data

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # RobotEnv methods
    # ----------------------------

    def _env_setup(self, initial_qpos):
        self.init_internal_vars(self.init_pos)
        self.set_init_pose()
        self.check_all_systems_ready()
        
    def init_internal_vars(self, init_pos_value):
        self.pos = [init_pos_value]
        self.joints = None
        
    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self._base_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _base_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_base_pub Publisher Connected")

        while (self._pole_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _pole_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_pole_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

    def _check_all_systems_ready(self, init=True):
        self.base_position = None
        while self.base_position is None and not rospy.is_shutdown():
            try:
                self.base_position = rospy.wait_for_message("/cartpole_v0/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current cartpole_v0/joint_states READY=>"+str(self.base_position))
                if init:
                    # We Check all the sensors are in their initial values
                    positions_ok = all(abs(i) <= 1.0e-02 for i in self.base_position.position)
                    velocity_ok = all(abs(i) <= 1.0e-02 for i in self.base_position.velocity)
                    efforts_ok = all(abs(i) <= 1.0e-01 for i in self.base_position.effort)
                    base_data_ok = positions_ok and velocity_ok and efforts_ok
                    rospy.logdebug("Checking Init Values Ok=>" + str(base_data_ok))
            except:
                rospy.logerr("Current cartpole_v0/joint_states not ready yet, retrying for getting joint_states")
        rospy.logdebug("ALL SYSTEMS READY")
        
            
    def move_joints(self, joints_array):
        joint_value = Float64()
        joint_value.data = joints_array[0]
        rospy.logdebug("Single Base JointsPos>>"+str(joint_value))
        self._base_pub.publish(joint_value)

        
    def get_clock_time(self):
        self.clock_time = None
        while self.clock_time is None and not rospy.is_shutdown():
            try:
                self.clock_time = rospy.wait_for_message("/clock", Clock, timeout=1.0)
                rospy.logdebug("Current clock_time READY=>" + str(self.clock_time))
            except:
                rospy.logdebug("Current clock_time not ready yet, retrying for getting Current clock_time")
        return self.clock_time
    
```

代码解释

```python
from gym import utils
from openai_ros.robot_envs import cartpole_env
from gym.envs.registration import register
from gym import error, spaces
import rospy
import math
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState
from rosgraph_msgs.msg import Clock
from openai_ros import robot_gazebo_env

# 注册CartPole环境到OpenAI Gym中
register(
    id='CartPoleStayUp-v0',  # 环境唯一标识符
    entry_point='openai_ros:CartPoleStayUpEnv',  # 环境的入口点
    max_episode_steps=1000,  # 每个episode的最大步数
)

class CartPoleEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    CartPole环境的ROS与Gazebo接口，用于在Gazebo中实现和训练CartPole任务。
    继承自RobotGazeboEnv，为CartPole任务提供必要的ROS话题和服务。
    """
    
    """
    This means that all the functions that are defined in the Gazebo Environment will also be accessible from this class.
    """
    def __init__(self, control_type):
        """
        初始化CartPole环境。
        参数:
        - control_type: 控制类型，包括 'velocity', 'position', 'effort'。
        """
        self.publishers_array = []
        self._base_pub = rospy.Publisher('/cartpole_v0/foot_joint_velocity_controller/command', Float64, queue_size=1)
        self._pole_pub = rospy.Publisher('/cartpole_v0/pole_joint_velocity_controller/command', Float64, queue_size=1)
        self.publishers_array.append(self._base_pub)
        self.publishers_array.append(self._pole_pub)
        
        rospy.Subscriber("/cartpole_v0/joint_states", JointState, self.joints_callback)
        
        self.control_type = control_type
        # 根据控制类型定义控制器列表
        if self.control_type == "velocity":
            self.controllers_list = [
                'joint_state_controller',
                'pole_joint_velocity_controller',
                'foot_joint_velocity_controller',
            ]
        elif self.control_type == "position":
            self.controllers_list = [
                'joint_state_controller',
                'pole_joint_position_controller',
                'foot_joint_position_controller',
            ]
        elif self.control_type == "effort":
            self.controllers_list = [
                'joint_state_controller',
                'pole_joint_effort_controller',
                'foot_joint_effort_controller',
            ]

        self.robot_name_space = "cartpole_v0"
        self.reset_controls = True

        self._seed()
        self.steps_beyond_done = None
        """
        作用: 调用基类 RobotGazeboEnv 的构造函数。
	参数说明:
	controllers_list: 控制器列表，根据 control_type 属性初始化，定义了控制CartPole的ROS控制器。
	robot_name_space: ROS的命名空间，用于指定节点和话题的范围。
	reset_controls: 布尔值，指示是否在每次环境重置时重置控制器。
	返回类型: 无返回值。这行代码通过调用基类的构造函数来完成环境对象的初始化，确保所有基类功能都被正确设置。	
        """
        super(CartPoleEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=self.reset_controls
        )

    def joints_callback(self, data):
        """
        ROS订阅回调函数，更新关节状态。
        """
        self.joints = data

    def _seed(self, seed=None):
        """
        设置环境的随机种子，保证实验的可复现性。
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def _env_setup(self, initial_qpos):
        """
        环境设置，在每次reset时调用，设置初始位置。
        """
        self.init_internal_vars(self.init_pos)
        self.set_init_pose()
        self.check_all_systems_ready()
        
    def init_internal_vars(self, init_pos_value):
        """
        初始化内部变量，包括初始位置。
        """
        self.pos = [init_pos_value]
        self.joints = None
        
    def check_publishers_connection(self):
        """
        检查所有发布者是否已连接，确保能发送命令到Gazebo。
        """
        rate = rospy.Rate(10)  # 10Hz的频率
        while (self._base_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No subscribers to _base_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("_base_pub Publisher Connected")

        while (self._pole_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No subscribers to _pole_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("_pole_pub Publisher Connected")
        rospy.logdebug("All Publishers READY")

    def _check_all_systems_ready(self, init=True):
        """
        确认所有系统就绪，可以开始发送和接收数据。
        """
        self.base_position = None
        while self.base_position is None and not rospy.is_shutdown():
            try:
                self.base_position = rospy.wait_for_message("/cartpole_v0/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current cartpole_v0/joint_states READY=>" + str(self.base_position))
                if init:
                    positions_ok = all(abs(i) <= 1.0e-02 for i in self.base_position.position)
                    velocity_ok = all(abs(i) <= 1.0e-02 for i in self.base_position.velocity)
                    efforts_ok = all(abs(i) <= 1.0e-01 for i in self.base_position.effort)
                    base_data_ok = positions_ok and velocity_ok and efforts_ok
                    rospy.logdebug("Checking Init Values Ok=>" + str(base_data_ok))
            except:
                rospy.logerr("Current cartpole_v0/joint_states not ready yet, retrying for getting joint_states")
        rospy.logdebug("ALL SYSTEMS READY")
        
    def move_joints(self, joints_array):
        """
        发布新的关节位置到Gazebo，驱动CartPole运动。
        """
        joint_value = Float64()
        joint_value.data = joints_array[0]
        rospy.logdebug("Single Base JointsPos>>" + str(joint_value))
        self._base_pub.publish(joint_value)

    def get_clock_time(self):
        """
        获取并返回仿真环境中的当前时钟时间。
        """
        self.clock_time = None
        while self.clock_time is None and not rospy.is_shutdown():
            try:
                self.clock_time = rospy.wait_for_message("/clock", Clock, timeout=1.0)
                rospy.logdebug("Current clock_time READY=>" + str(self.clock_time))
            except:
                rospy.logdebug("Current clock_time not ready yet, retrying for getting Current clock_time")
        return self.clock_time

```

It's very important that you understand that you can add all the functions that you want to this Robot Environment. However, there are some functions that will **ALWAYS** need to be defined here because they are required for the rest of the Environments to work properly. These functions include:

- The **__init__** function
- The **_check_all_systems_ready()** function

## Gazebo Environment

The Gazebo Environment is used to connect the Robot Environment to the Gazebo simulator. For instance, it takes care of the resets of the simulator after each step, or the resets of the controllers (if needed).

The most important thing you need to know about this environment is that it **will be transparent** to you. And what does this mean? Well, it basically means that you don't have to worry about it. This environment **will always be the same**, regardless of the robot or the kind of task to solve. So, you won't have to change it or work over it. Good news, right?

Here you can have a look at how this environment looks:

**robot_gazebo_env.py**



```python
import rospy
import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection
#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from theconstruct_msgs.msg import RLExperimentInfo

# https://github.com/openai/gym/blob/master/gym/core.py
class RobotGazeboEnv(gym.Env):

    def __init__(self, robot_name_space, controllers_list, reset_controls):

        # To reset Simulations
        self.gazebo = GazeboConnection()
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()

        # Set up ROS related variables
        self.episode_num = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)

    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        self.gazebo.unpauseSim()
        self._set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        self._publish_reward_topic(reward, self.episode_num)

        return obs, reward, done, info

    def reset(self):
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Increases the episode number by one
        :return:
        """
        self.episode_num += 1

    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation
        """
        if self.reset_controls :
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
            
        else:
            self.gazebo.unpauseSim()
            
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
        

        return True

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()
```

代码解释



```python
import rospy
import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection
from theconstruct_msgs.msg import RLExperimentInfo

# 继承自 gym.Env，创建自定义的用于 Gazebo 机器人仿真的环境类
class RobotGazeboEnv(gym.Env):

    def __init__(self, robot_name_space, controllers_list, reset_controls):
        """
        初始化 RobotGazeboEnv 环境。

        参数:
        - robot_name_space (str): 该机器人环境的 ROS 命名空间。
        - controllers_list (list): 要管理的控制器名称列表。
        - reset_controls (bool): 指示在环境重置时是否重置控制器的标志。

        此设置包括与 Gazebo 和 ROS 控制器的连接，对于仿真至关重要。
        """
        self.gazebo = GazeboConnection()  # 管理与 Gazebo 仿真的连接。
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)  # 处理 ROS 控制器。
        self.reset_controls = reset_controls  # 确定是否需要重置仿真控制器。
        self.seed()  # 初始化随机种子。

        # ROS 发布者和订阅者
        self.episode_num = 0  # 跟踪周期数。
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)  # 发布奖励以便监控。

    def seed(self, seed=None):
        """
        为这个环境的随机数生成器设置种子，以确保实验的可复现性。
        
        参数:
        - seed (int, 可选): 用于随机数生成的种子。
        
        返回:
        - list: 包含使用的种子值的列表。
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        执行环境动力学的一个时间步。
        
        参数:
        - action: 环境代理提供的动作。
        
        返回:
        - obs: 代理对当前环境的观察。
        - reward: 上一个动作后返回的奖励量。
        - done: 该周期是否已结束，此时进一步的 step() 调用将返回未定义结果。
        - info: 包含辅助诊断信息（有助于调试和有时候是学习）。
        """
        self.gazebo.unpauseSim()  # 解除仿真暂停以进行观察。
        self._set_action(action)  # 应用动作。
        self.gazebo.pauseSim()  # 动作应用后暂停仿真。
        obs = self._get_obs()  # 获取当前可观测的状态。
        done = self._is_done(obs)  # 检查该周期是否结束。
        reward = self._compute_reward(obs, done)  # 计算奖励。
        self._publish_reward_topic(reward, self.episode_num)  # 发布奖励信息。
        return obs, reward, done, {}

    def reset(self):
        """
        重置环境状态并返回初始观察。
        
        返回:
        - obs: 重置环境后的初始观察。
        """
        rospy.logdebug("重置 RobotGazeboEnvironment")
        self._reset_sim()  # 重置仿真。
        self._init_env_variables()  # 初始化环境变量。
        self._update_episode()  # 更新周期计数器。
        return self._get_obs()  # 返回初始观察。

    def close(self):
        """
        执行必要的清理操作。
        """
        rospy.logdebug("关闭 RobotGazeboEnvironment")
        rospy.signal_shutdown("关闭 RobotGazeboEnvironment")  # 关闭 ROS 节点。

    def _update_episode(self):
        """
        增加一次周期计数。
        """
        self.episode_num += 1

    def _publish_reward_topic(self, reward, episode_number=1):
        """
        将给定的奖励发布到 ROS 主题中。
        
        参数:
        - reward: 要发布的奖励。
        - episode_number: 当前周期数。
        """
        reward_msg = RLExperimentInfo()  # 创建用于发布的消息。
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)  # 发布奖励。

    # 子类应该实现的方法
    def _reset_sim(self):
        """
        重置仿真至初始状态。
        """
        if self.reset_controls:
            self.gazebo.unpauseSim()  # 解除仿真暂停以进行重置。
            self.controllers_object.reset_controllers()  # 将控制器重置至初始状态。
            self._check_all_systems_ready()  # 检查所有系统是否就绪。
            self._set_init_pose()  # 设置初始姿势。
            self.gazebo.pauseSim()  # 设置姿势后暂停仿真。
            self.gazebo.resetSim()  # 重置仿真以开始新的。
            self.gazebo.unpauseSim()  # 解除暂停以重新应用设置。
            self.controllers_object.reset_controllers()  # 仿真重置后再次重置控制器。
            self._check_all_systems_ready()  # 最终检查。
            self.gazebo.pauseSim()  # 最终暂停以开始处理。
        else:
            self.gazebo.unpauseSim()  # 如果不重置控制器，只解除暂停。
            self._check_all_systems_ready()  # 检查系统就绪情况。
            self._set_init_pose()  # 设置初始机器人姿势。
            self.gazebo.pauseSim()  # 设置后暂停。
            self.gazebo.resetSim()  # 重置仿真。
            self.gazebo.unpauseSim()  # 解除暂停以应用重置设置。
            self._check_all_systems_ready()  # 最终系统检查。
            self.gazebo.pauseSim()  # 最终暂停前准备。

        return True

    def _set_init_pose(self):
        """
        设置机器人的初始姿势。
        """
        raise NotImplementedError("该方法应由子类重写")

    def _check_all_systems_ready(self):
        """
        确保所有的传感器、发布者和其他仿真系统处于运行状态。
        """
        raise NotImplementedError("该方法应由子类重写")

    def _get_obs(self):
        """
        获取当前的观测值。
        """
        raise NotImplementedError("该方法应由子类重写")

    def _init_env_variables(self):
        """
        在每个周期开始时初始化需要的环境变量。
        """
        raise NotImplementedError("该方法应由子类重写")

    def _set_action(self, action):
        """
        将给定的动作应用于仿真。
        """
        raise NotImplementedError("该方法应由子类重写")

    def _is_done(self, observations):
        """
        根据观测值判断周期是否结束（例如机器人是否倒下）。
        """
        raise NotImplementedError("该方法应由子类重写")

    def _compute_reward(self, observations, done):
        """
        根据给定的观测值和完成状态计算奖励。
        """
        raise NotImplementedError("该方法应由子类重写")

    def _env_setup(self, initial_qpos):
        """
        环境的初始配置。可以用于配置初始状态和提取仿真数据。
        """
        raise NotImplementedError("该方法应由子类重写")

```

## Visualizing the reward

As you have seen in the previous section, where we discussed about the Gazebo Environment, this environment publishes the reward into a topic named **/openai/reward**. This is very important, because it will allow us to visualize our reward using regular ROS tools. For instance, **rqt_multiplot**.

**rqt_multiplot** is similar to rqt_plot, since it also provides a GUI for visualizing 2D plots, but it's a little bit more complex and provides more features. For instance, you can visualize multiple plots at the same time, or you can also customize the axis for your plot.

In the next exercise, we are going to see how to visualize the reward of our training using the rqt_multiplot tool.

### **Exercise 2.6**

a) First of all, let's start the rqt_multiplot tool.

Execute in WebShell #1

```
rosrun rqt_multiplot rqt_multiplot
```

b) Hit the icon with a screen in the top-right corner of the IDE window![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/font-awesome_desktop.png)in order to open the Graphic Interface window.

You should now see something like this in this new window.

![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/rqt_multiplot1.png)

c) Next, let's launch our training script, so that we start getting the rewards published into the **/openai/reward** topic.

Execute in WebShell #1



```
roslaunch cartpole_v0_training start_training.launch
```

d) Next, let's configure our plot. For that, first, you'll need to click on the **Configure plot** button, at the top-right side of the window.

![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/rqt_multiplot2.png)

e) Give a name to your new plot, and click on the **Add curve** icon.

![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/rqt_multiplot3.png)

f) Next, set the topic to **/openai/reward**. Automatically, the rest of the field will be autocompleted. For the X-Axis, select the **episode_number** field. For the Y-Axis, select the **episode_reward** field. Like in the picture below.

![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/rqt_multiplot4.png)

When you are done, just click on the **Enter** key on your keyboard.

g) Finally, in the plot visualization window, you will need to click on the **Run plot** button in order to start visualizing your new plot.

![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/rqt_multiplot5.png)

If everything went OK, you should visualize something like this:

![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/rqt_multiplot7.png)

You can also save, if you want, your plot configuration. This way, you will be able to load it at any time later. Remember to save it into your **catkin_ws/src** workspace, so that it will also be saved when you close the Course.

![img](https://s3.eu-west-1.amazonaws.com/notebooks.ws/course_openai_ROS/img/rqt_multiplot6.png)



## Unit 2: How to apply openai_ros to a new robot

**The moving cube robot**

This robot was developed by the **ETHZurich** and it's a perfect platform for control theory and mechanical physics. You can find more information in this link: [Cubli Research](http://www.idsc.ethz.ch/research-dandrea/research-projects/cubli.html)

In order to see the video below, select the next cell of the notebook and press the *play* button.



```python
from IPython.display import YouTubeVideo
# Cubli Robot created By ETHZurich
# Video credit: William Stein.
YouTubeVideo('n_6p-1J551Y')
```

**The simulation of the robot**

We have created a simulation of the Cubli robot, the one you can see on the simulation window. This first simulated version has only **One** inertia disk, but it s enough to the goal we want to achieve.

我们已经对Cubli机器人进行了模拟，您可以在模拟窗口上看到一个。这个第一个模拟版本只有一个惯性磁盘，但足以实现目标。



**The goal we want the robot to learn**

We want to make this robot be able to walk around in the direction that we want. And it has to do it by learning by itself, not by using mathematical calculations as was done with the original.

So, the objective is to make the Cubli robot learn how to **move forwards in the WORLD Y-AXIS direction.**

In the next units of this course, we are going to learn how to build all the software parts to achieve that goal

**Structure of the next units:**

1. On unit 3, you are going to **create a new Robot Environment for the Cubli** that allows you to access the sensors and actuators.
2. On unit 4, you are going to **create the Training Environment** that inherits from the Robot Environment that you created. You will use this Environment to define the reward and the conditions to detect when the task is done. You will also use it to **provide to the training algorithm the vector of observations**, as well as to **provide to the Robot Environment the actual commands to move** based on the action decided by the training algorithm.
3. On unit 5, you will **create a Training Script for Qlearning and deepQ** that uses your created Task Environment.

**我们希望机器人实现的目标**

我们的目标是让这个机器人能够自主学习在我们希望的方向上行走。它需要通过自学来实现这一点，而不是使用像原来那样的数学计算。

因此，目标是让Cubli机器人学会如何**在世界Y轴方向上前进**。

在接下来的课程单元中，**我们将学习构建实现该目标所需的所有软件部分**

**接下来单元的结构**：

1. 在第三单元，你将**为Cubli创建一个新的机器人环境**，使你能够访问传感器和执行器。
2. 在第四单元，你将**创建一个继承自你创建的机器人环境的训练环境**。你将使用这个环境来定义奖励和检测任务完成的条件。你还将使用它来**提供给训练算法观察向量**，以及根据训练算法决定的动作**向机器人环境提供实际的移动命令**。
3. 在第五单元，你将**为Q学习和深度Q创建一个训练脚本**，使用你创建的任务环境。



So, you now understand how the **CartPole Environment structure** works. Now, we need to create one from scratch for the One-Disk-Moving-Cube. There are three main steps:

- We need to create a **Robot Environment** that has the basic functions needed to use the RoboCube.
- We need to decide how we move the RoboCube and how to get the sensor data.
- We need to create functions that allow environments that inherit from our class to retrieve sensor data and access the cube's functionality, without knowing all the ROS related stuff.



With this created, you will have a **Robot Environment** that can be used by any **Task Environment** that you create afterwards.



### Step 0. Create a package to place all your code in

We will first create a package that will hold all the code that you generate in the next chapters.

Execute in WebShell #1

```
roscd; cd ../src; 
```



```
catkin_create_pkg my_moving_cube_pkg rospy openai_ros
```



```
cd my_moving_cube_pkg/
```



```
mkdir scripts; cd scripts
```



## Step 1. Create a basic Robot Environment

To create our Robot Environment, we will start from a Robot Environment template (see the code below). You can also find this template inside the [*openai_ros* package](https://bitbucket.org/theconstructcore/openai_ros/src/version2/) under the *openai_ros/templates* directory.

**template_my_robot_env.py**



```python
from openai_ros import robot_gazebo_env


class MyRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = ['my_robot_controller1','my_robot_controller2', ..., 'my_robot_controllerX']

        self.robot_name_space = "my_robot_namespace"

        reset_controls_bool = True or False
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        
        super(MyRobotEnv, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=reset_controls_bool)

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO
        return True
    
    # Methods that the TaskEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TaskEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TaskEnvironment will need.
    # ----------------------------
```





You can see here that the template is divided into four parts:

1. Initialization of the Robot Environment class (in the example above, **MyRobotEnv**). It inherits from the *RobotGazeboEnv*.
2. Definition of the virtual methods needed by the Gazebo Environment, which were declared virtually inside *RobotGazeboEnv*.
3. Virtual definition of methods that the Task Environment will need to define here as virtual because they will be used in the RobotGazeboEnv GrandParentClass and defined in the Task Environment.
4. Definition of methods that the Task Environment will need to use from the class.



```python
from openai_ros import robot_gazebo_env

class MyRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """所有机器人环境的超类。"""
    
    def __init__(self):
        """
        初始化一个新的机器人环境。
        此构造函数设置必要的变量，并调用父类的初始化函数。
        """
        # 通过构造函数给出的变量

        # 内部变量
        self.controllers_list = ['my_robot_controller1', 'my_robot_controller2', ..., 'my_robot_controllerX']
        # 控制器列表，包含管理机器人各部分的控制器名称。

        self.robot_name_space = "my_robot_namespace"
        # 机器人的ROS命名空间，用于区分不同机器人或相同机器人的不同实例。

        reset_controls_bool = True or False
        # 布尔值，指示是否在环境重置时重置控制器。
        
        # 调用父类的初始化函数
        super(MyRobotEnv, self).__init__(controllers_list=self.controllers_list,
                                         robot_name_space=self.robot_name_space,
                                         reset_controls=reset_controls_bool)

    def _check_all_systems_ready(self):
        """
        检查所有传感器、发布者和其他仿真系统是否处于运行状态。
        返回：
        - True: 表示所有系统都已准备好。
        """
        # TODO: 实现具体检查逻辑
        return True
    
    def _set_init_pose(self):
        """
        设置机器人的初始姿态。
        此方法必须在子类中实现。
        """
        raise NotImplementedError("必须在子类中实现此方法。")
    
    def _init_env_variables(self):
        """
        在每个周期开始时初始化所需的环境变量。
        此方法必须在子类中实现。
        """
        raise NotImplementedError("必须在子类中实现此方法。")

    def _compute_reward(self, observations, done):
        """
        根据给定的观察和完成状态计算奖励。
        参数:
        - observations: 当前的观察数据。
        - done: 布尔值，表示周期是否结束。
        此方法必须在子类中实现。
        """
        raise NotImplementedError("必须在子类中实现此方法。")

    def _set_action(self, action):
        """
        将给定的动作应用于仿真。
        参数:
        - action: 要应用的动作。
        此方法必须在子类中实现。
        """
        raise NotImplementedError("必须在子类中实现此方法。")

    def _get_obs(self):
        """
        返回当前的观察数据。
        此方法必须在子类中实现。
        """
        raise NotImplementedError("必须在子类中实现此方法。")

    def _is_done(self, observations):
        """
        根据给定的观察数据判断周期是否结束。
        参数:
        - observations: 当前的观察数据。
        此方法必须在子类中实现。
        """
        raise NotImplementedError("必须在子类中实现此方法。")

```





## Step 2- Initialization of the class MyRobotEnv

Open the *my_cube_single_disk_env.py* file on the IDE and copy the following text inside:

```python
#! /usr/bin/env python

from openai_ros import robot_gazebo_env

class MyCubeSingleDiskEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, init_roll_vel):
        """Initializes a new CubeSingleDisk environment.
        """
        # Variables that we give through the constructor.
        self.init_roll_vel = init_roll_vel

        self.controllers_list = ['my_robot_controller1','my_robot_controller2', ..., 'my_robot_controllerX']

        self.robot_name_space = "my_robot_namespace"

        reset_controls_bool = True or False
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(MyCubeSingleDiskEnv, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=reset_controls_bool)
```

In the previous code, we are creating the class for the Cubli Robot Environment. We are calling that class: **MyCubeSingleDiskEnv**.

As you can see, we just took the first lines of the template above and modified a few things to accomodate for the Cubli robot. Let's review them:

Here we import the **robot_gazebo_env** from the python module folder openai_ros. Inside **robot_gazebo_env**, we find the class **RobotGazeboEnv**. In the class definition of **MyCubeSingleDiskEnv**, we give inheritance to **robot_gazebo_env.RobotGazeboEnv**. That means to look for the python module file **robot_gazebo_env**, and inside, get the class **RobotGazeboEnv**.

In the previous code, we are creating the class for the Cubli Robot Environment. We are calling that class: **MyCubeSingleDiskEnv**.

As you can see, we just took the first lines of the template above and modified a few things to accomodate for the Cubli robot. Let's review them:

Here we import the **robot_gazebo_env** from the python module folder openai_ros. Inside **robot_gazebo_env**, we find the class **RobotGazeboEnv**. In the class definition of **MyCubeSingleDiskEnv**, we give inheritance to **robot_gazebo_env.RobotGazeboEnv**. That means to look for the python module file **robot_gazebo_env**, and inside, get the class **RobotGazeboEnv**.

















































