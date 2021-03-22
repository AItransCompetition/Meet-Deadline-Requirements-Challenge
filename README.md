# [ACM Multimedia 2021 Grand Challenge: Meet Deadline Requirements](https://www.aitrans.online/MMGC2021/) <!-- omit in toc -->

![running demos](https://github.com/AItransCompetition/Meet-Deadline-Requirements-Challenge/actions/workflows/main.yml/badge.svg)

This repo contains demos and datasets of *ACM Multimedia 2021 Grand Challenge: Meet Deadline Requirements*, you can see more information about this challenge on our [website](https://www.aitrans.online/MMGC2021/). Any registration and participation are welcome.

- [Quick Start](#quick-start)
  - [Requirements](#requirements)
  - [Run A Demo](#run-a-demo)
- [Your Task](#your-task)
  - [Blocks Scheduler](#blocks-scheduler)
  - [Bandwidth Estimator](#bandwidth-estimator)
  - [Other States Update Points](#other-states-update-points)
- [Solution Evaluation](#solution-evaluation)
  - [Create An Emulator](#create-an-emulator)
  - [Run An Emulator](#run-an-emulator)
  - [cal_qoe](#cal_qoe)
- [Analyzing Tools](#analyzing-tools)
  - [print_debug](#print_debug)
  - [analyze_emulator](#analyze_emulator)
  - [plot_rate](#plot_rate)

# Quick Start

## Requirements

Basic requirements: You can install our emulator platform([simple-emulator](https://pypi.org/project/simple-emulator/)) and basic packages needed by the emulator using:   
```bash
pip3 install -r requirements.txt
```

Other requirements: If you want to use a learning-based solution, you need to install some other packages. For example, you need to install [torch](https://pypi.org/project/torch/) to run [RL-demo-using-torch](https://github.com/AItransCompetition/Meet-Deadline-Requirements/blob/master/solution_demos/rl_torch/solution.py).  
(*The packages supported on our submission evaluating system can be found in [import-package](https://github.com/AItransCompetition/Meet-Deadline-Requirements/tree/master/solution_demos#import-package).)

## Run A Demo
After basic requirements installed, you can run run_this.py to evaluate the reno solution demo:
```bash
python3 run_this.py
```
Now you have successfully run the demo [Reno](https://github.com/AItransCompetition/Meet-Deadline-Requirements/tree/master/solution_demos#reno), which can be used directly in the [submitting](https://github.com/AItransCompetition/Meet-Deadline-Requirements/tree/master/solution_demos#submit).   
If you want to change the evaluation process, you can modify run_this.py or create your own evaluating program. [Solution Evaluation](#solution-evaluation) will show you more details on how to achieve this.


# Your Task 

Participants need to implement a *Scheduler*. The *Scheduler* consists of two parts-*Blocks Scheduler* and *Bandwidth Estimator*.
We have provided some [demos of solution](https://github.com/AItransCompetition/Meet-Deadline-Requirements/tree/master/solution_demos).

## Blocks Scheduler

Select which block in `block_queue` should be sent at the time `cur_time`. You need to implement your scheduler algorithm in method `select_block`. 

The emulator will call `select_block` every time sender tries to send a packet from a block. Your algorithm need to select a block and return its id.

**Input** of the blocks scheduler: `select_block` will get `cur_time` which means current time, and `block_queue`:  

- cur_time

  > The time when the packet arrived.

- block_queue

  > This parameter is a list object composed of currently sendable blocks. It is recommended to read about its detailed description : [block_queue](https://github.com/AItransCompetition/simple_emulator/tree/mmgc#table--block_queue)

**Output** of the blocks scheduler: Just return the index of the block to be sent in the `block_queue` list, like `0` means that sending the first block in `block_queue`.

## Bandwidth Estimator

Update the sending rate of packets. You need to implement your bandwidth estimator algorithm in method  `cc_trigger`.

The emulator will call `cc_trigger` every time packet acked  or packet dropped event happen. And you can update the sending rate and states of your algorithm when `cc_trigger` called.  

**Input** of the estimator: `cc_trigger` will get `cur_time` which means current time, and a two-tuple `event_info`. You can use these signals in your estimating.

- cur_time

  > The time when the packet arrived.

- event_info

  - event_type

    > We divide the packet into three categories : PACKET_TYPE_FINISHED, PACKET_TYPE_TEMP, PACKET_TYPE_DROP.
    >
    > PACKET_TYPE_FINISHED : The acknowledge packet that successfully reached the source point;
    >
    > PACKET_TYPE_TEMP : The packet that have not yet reached the source point;
    >
    > PACKET_TYPE_DROP : The packet used to inform the source point of packet loss.

  - packet_information_dict

    > The packet it the object implemented in "objects/packet.py". But we recommend you to get more information at [packet_information_dict](https://github.com/AItransCompetition/simple_emulator#table--packet_information_dict).

**Output** of the estimator: in addition to controlling by the estimated bandwidth, we also support solutions based on congestion window. The return of `cc_trigger` consists of `cwnd` and `send_rate`. E.g
```python
{
    "cwnd" : 10,
    "send_rate" : 10
}
```

## Other States Update Points

You can set or update the states of your algorithm in methods `__init__` and `on_packet_sent`. 
Moreover, you can also update the send rate or CWND when returning from `on_packet_sent`. Emulator calls `on_packet_sent` every time sender tries to send a packet.  

# Solution Evaluation

## Create An Emulator

```python
# Instantiate the solution
my_solution = solution.MySolution()

# Create emulator
emulator = Emulator(
    block_file=["datasets/application_traces/data_video.csv", "datasets/application_traces/data_audio.csv"],
    trace_file="datasets/network_traces/trace.txt",
    solution=my_solution,
    # enable logging packet. You can train faster if USE_CWND=False
    ENABLE_LOG=True
)
```

The `block_file`, `trace_file`, `solution` are necessary to construct an Emulator. The `block_file` list contains one or more application trace files.  The `trace_file` represents the network trace file. The `solution` contains methods you implemented.

The specific meanings of the optional parameters are as follows: 

| Parameter Name            | Default Value | Description                                                     |
| ------------------- | ------ | ------------------------------------------------------------ |
| USE_CWND            | False   | Whether the algorithm is based on the congestion window |
| ENABLE_DEBUG        | False  | Output DEBUG information                       |
| ENABLE_LOG          | True   | Output packet-level LOG           |
| MAX_PACKET_LOG_ROWS | 4000   | The maximum number of records in a packet-level log file, if it exceeds the maximum number of records, a new file will be created |
| SEED                | -      | Emulator random seed                         |
| RUN_DIR             | -      | Eimulator run root directory                |

## Run An Emulator

```python
emulator.run_for_dur(15)
```

The optional parameter indicates the cut-off time (in seconds) of the emulation. The default is infinite, that is, the emulation will not end until application trace ends.

## cal_qoe

```python
print("Qoe : %d" % (cal_qoe()) )
```

Calculate the QoE score based on the `output/block.log` in the running result.

# Analyzing Tools

In order to help participants better optimize the solution, we also provide some functional tools in the emulator.

```python
emulator.print_debug()

analyze_emulator(log_packet_file, file_range="all")

plot_rate(log_packet_file, trace_file="datasets/network_traces/trace.txt", file_range="all", sender=[1])
```

## print_debug

Show a brief analysis result, including delay, packet loss, etc. The output example and corresponding explanations are as follows.

```
---Link Debug---
Link: 1								# Link ID
Bandwidth: 1333.333333				# Link bandwidth
Delay: 0.001000						# Link propagation delay
Queue Delay: 0.003000				# Queuing delay of the last packet in this link
Max Queue Delay: 0.041250			# Maximum queuing delay in this link
One Packet Queue Delay: 0.000750	# Queuing delay of a single packet in this link
Link: 2
Bandwidth: inf
Delay: 0.001000
Queue Delay: 0.000000
Max Queue Delay: 0.000000
One Packet Queue Delay: 0.000000
---Sender Debug---
Sender: 1							# Sender ID
Rate: inf							# Sender's sending rate
Sent: 18247							# Number of packet sent
Acked: 18184						# Number of packet acked
Lost: 61							# Number of packet lost
Min Latency: 0.002					# Minimum queuing delay in acked packets
```

## analyze_emulator

The delay curve of the tested algorithm during the evaluation process.

![image-20210315204252305](https://www.aitrans.online/static/MMGC2021/image-20210315204252305.png)

The required parameter is `log_file`, which represents the packet-level log file that needs to be analyzed.

The optional parameters are as follows:

| Parameters | Defaults | Explanation                                                  |
| ---------- | -------- | ------------------------------------------------------------ |
| rows       | None     | Limit the number of log lines that will be read              |
| trace_file | None     | The path of network trace file.Use this option to draw bandwidth and statistical throughput in the same picture |
| time_range | None     | Log file time range used                                     |
| scatter    | False    | Whether to switch to scatter chart (default is polyline)     |
| file_range | None     | Log file range. For multiple file logs, you can use this item to specify the range (for example: 'file_range=[1,3]' means files from 1 to 3 (left closed and right open interval), "all" means all files) |
| sender     | None     | Filter log file by sender ID                                 |

## plot_rate

The transmission speed change curve during the evaluation progress.

![image-20210315204059497](https://www.aitrans.online/static/MMGC2021/image-20210315204059497.png)

The required parameter is **log_file**, which represents the packet-level log file that needs to be analyzed.

The optional parameters are as follows:

| Parameters | Defaults | Explanation                                                  |
| ---------- | -------- | ------------------------------------------------------------ |
| rows       | None     | Limit the number of log lines that will be read              |
| trace_file | None     | The path of network trace file.Use this option to draw bandwidth and statistical throughput in the same picture |
| time_range | None     | Log file time range used                                     |
| scatter    | False    | Whether to switch to scatter chart (default is polyline)     |
| file_range | None     | Log file range. For multiple file logs, you can use this item to specify the range (for example: 'file_range=[1,3]' means files from 1 to 3 (left closed and right open interval), "all" means all files) |
| sender     | None     | Filter log file by sender ID                                 |
| size       | 1        | Draw a data point every size seconds                         |
