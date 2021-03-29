# Solution

## Submit
* The type of file you submit is "*.zip". The name is like "submit.zip".
* The "submit.zip" is gotten by compressing a folder called "submit". The path is like following:
```
submit
│   solution.py
│   your_other_file(README.md, your_model.pb, sub_module.py.etc)
|
```

- For participants who want to use local files or models, it is recommended to use the following method to obtain the current directory, and then add a relative path to import the trained model.

```python
import os, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
model_path = current_dir+"/model.path"
```
- Submission limit: 3 times a day during the competition.

## Demos Of solution.py

We provide several simple solutions as demos, all of which can be used directly in the submission. 

### Reno

In this demo, the `cc_trigger` is the implementation of [Reno](https://en.wikipedia.org/wiki/TCP_congestion_control). The `select_block` selects the best block based on block creation time and deadline factors.

### RL_torch

In this demo, the `select_block` is the same as that in the demo [Reno](#Reno). 

The `cc_trigger` contains a bandwidth estimator based on reinforcement learning, which is implemented by [PyTorch](https://github.com/pytorch/pytorch). 

In the bandwidth estimator, the solution first defines a 3-layer fully connected network object —— neural network, each of which has a size of N_STATES, N_HIDDEN, and N_ACTIONS. Based on the neural network, the solution implements [DQN Algorithm](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

### RL_tensorflow

As the [RL_torch](#RL_torch) does, this demo also contains a bandwidth estimator based on reinforcement learning but implemented by [TensorFlow](https://github.com/tensorflow/tensorflow).  
The `select_block` is the same as that in the demo [Reno](#Reno).

## Import Package

The official submission system has provided some pre-installed libraries. The following is the pre-installed library name and version number information：

| Library name | torch | tensorflow | numpy | simple_emulator |
| :----------: | :---: | :--------: | :---: | :-------------: |
|   Version    |  1.8.0  |    2.4.1     | 1.19.5  |       0.0.5       |

* If you want to add some site packages, please contact us.
