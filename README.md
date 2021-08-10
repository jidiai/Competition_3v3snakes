# Competition_3v3snakes

### Environment

<!-- ![image](https://github.com/jidiai/Competition_3v3snakes/blob/master/assets/snakesdemo.gif) -->
<img src="https://github.com/jidiai/Competition_3v3snakes/blob/master/assets/snakesdemo.gif" alt="Competition_3v3snakes" width="500" height="250" align="middle" />

Check details in Jidi Competition [RLChina2021智能体竞赛](http://www.jidiai.cn/compete_detail?compete=6)

---
### Dependency
You need to create competition environment.
>conda create -n snake1v1 python=3.6

>conda activate snake1v1

>pip install -r requirements.txt

If your device is Mac M1, try the command below 
>conda install pytorch torchvision matplotlib gym tensorboardX pillow numpy tabulate pyyaml

---
### How to train rl-agent

>python rl_trainer/main.py

By default-parameters, the total reward of training is shown below.

![image](https://github.com/jidiai/Competition_3v3snakes/blob/master/assets/training.png)


You can edit different parameters, for example

>python rl_trainer/main.py --algo "bicnet" --epsilon 0.8

Baseline performance:

You can locally evaluation your model.

>python evaluation_local.py --my_ai rl --opponent random

![image](https://github.com/jidiai/Competition_3v3snakes/blob/master/assets/baseline.png)


---
### How to test submission 
You can locally test your submission. At Jidi platform, we evaluate your submission as same as **run_log.py**

Once you run this file, you can locally check battle logs in the folder named "logs".

For example, 
>python run_log.py --my_ai "random" --opponent "rl"

---
### Ready to submit 

1. Random policy --> **agent/random/submission.py**
2. RL policy --> **all files in agent/rl/**

---
### Watch reply locally

1. Open reply/reply.html in any browser.
2. Load a log.
3. Reply and watch ^0^.



[comment]: <> (## Content)

[comment]: <> (```)

[comment]: <> (|-- Competition_3v3snakes           // https://github.com/jidiai/Competition_3v3snakes.git)

[comment]: <> (    |-- env		                    // game environments)

[comment]: <> (    |	|-- obs_interfaces		    // Super Class)

[comment]: <> (	|	|	|-- observation.py		// support Grid interface)

[comment]: <> (	|	|-- simulators		        // Super Class)

[comment]: <> (	|	|	|-- game.py)

[comment]: <> (	|	|	|-- gridgame.py         )

[comment]: <> (	|	|-- config.ini		        // env config)

[comment]: <> (	|	|-- chooseenv.py )

[comment]: <> (	|	|-- snakes.py)

[comment]: <> (	|-- examples)

[comment]: <> (	|   |-- random                  // random policy)

[comment]: <> (	|   |   |-- submission.py       // you can submit this file to the platform without any modification)

[comment]: <> (	|   |-- myagent                 // customize policy)

[comment]: <> (	|   |   |-- dqn.py              )

[comment]: <> (	|   |   |-- submission.py       // main file，which should contain function named `my_controller`)

[comment]: <> (	|-- utils                       )

[comment]: <> (	|-- run_log.py		            // run an episode and generate .json log)

[comment]: <> (	|-- replay	                    // replay local util. Click replay.html and upload the json generated by run_log.py to replay )

[comment]: <> (```)







