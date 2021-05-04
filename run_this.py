"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""

from simple_emulator import SimpleEmulator, create_emulator

# We provided some function of plotting to make you analyze result easily in utils.py
from simple_emulator import analyze_emulator, plot_rate
from simple_emulator import constant

from simple_emulator import cal_qoe


def run_and_plot(emulator, network_trace, log_packet_file):
    # Run the emulator and you can specify the time for the emualtor's running.
    # It will run until there is no packet can sent by default.
    emulator.run_for_dur(15)

    # print the debug information of links and senders
    emulator.print_debug()

    # Output the picture of emulator-analysis.png
    # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#emulator-analysispng.
    analyze_emulator(log_packet_file, file_range="all", sender=[1])

    # Output the picture of rate_changing.png
    # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#cwnd_changingpng
    plot_rate(log_packet_file, trace_file=network_trace, file_range="all", sender=[1])

    print("Qoe : %d" % (cal_qoe()) )


def evaluate(solution_file, block_traces, network_trace, log_packet_file, second_block_file=None):
    # fixed random seed
    import random
    random.seed(1)

    # import the solution
    import importlib
    solution = importlib.import_module(solution_file)

    # Use the object you created above
    my_solution = solution.MySolution()

    # Create the emulator using your solution
    # Set second_block_file=None if you want to evaluate your solution in situation of single flow
    # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
    # You can get more information about parameters at https://github.com/AItransCompetition/simple_emulator/tree/master#constant
    emulator = create_emulator(
        block_file=block_traces,
        second_block_file=second_block_file,
        trace_file=network_trace,
        solution=my_solution,
        # enable logging packet. You can train faster if ENABLE_LOG=False
        ENABLE_LOG=True
    )
    run_and_plot(emulator, network_trace, log_packet_file)


if __name__ == '__main__':
    block_traces = ["datasets/scenario_2/blocks/block_video.csv", "datasets/scenario_2/blocks/block_audio.csv"]
    network_trace = "datasets/scenario_2/networks/traces_7.txt"
    
    # The file path of packets' log
    log_packet_file = "output/packet_log/packet-0.log"
    # Select the solution file
    solution_file = 'solution_demos.reno.solution'


    # The first sender will use your solution, while the second sender will send the background traffic

    # The block files for the first sender
    first_block_file = block_traces
    # The block files for the second sender
    second_block_file = ["datasets/background_traffic_traces/web.csv"]
    # Create the emulator and evaluate your solution
    evaluate(solution_file, first_block_file, network_trace, log_packet_file, second_block_file=second_block_file)

    # If only one block_traces is given, it means there will be no background traffic
    # evaluate(solution_file, block_traces, network_trace, log_packet_file)


