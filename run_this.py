"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""

from simple_emulator import SimpleEmulator

# We provided some function of plotting to make you analyze result easily in utils.py
from simple_emulator import analyze_emulator, plot_rate
from simple_emulator import constant

from simple_emulator import cal_qoe

def evaluate(solution_file, block_traces, network_trace, log_packet_file):
    # fixed random seed
    import random
    random.seed(1)

    # import the solution
    import importlib
    solution = importlib.import_module(solution_file)

    # Use the object you created above
    my_solution = solution.MySolution()

    # Create the emulator using your solution
    # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=False by default.
    # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
    # You can get more information about parameters at https://github.com/AItransCompetition/simple_emulator/tree/master#constant
    emulator = SimpleEmulator(
        block_file=block_traces,
        trace_file=network_trace,
        solution=my_solution,
        # enable logging packet. You can train faster if USE_CWND=False
        ENABLE_LOG=True
    )

    # Run the emulator and you can specify the time for the emualtor's running.
    # It will run until there is no packet can sent by default.
    emulator.run_for_dur(15)

    # print the debug information of links and senders
    emulator.print_debug()

    # Output the picture of emulator-analysis.png
    # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#emulator-analysispng.
    analyze_emulator(log_packet_file, file_range="all")

    # Output the picture of rate_changing.png
    # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#cwnd_changingpng
    plot_rate(log_packet_file, trace_file=network_trace, file_range="all", sender=[1])

    print("Qoe : %d" % (cal_qoe()) )


if __name__ == '__main__':
    block_traces = ["datasets/application_traces/data_video.csv", "datasets/application_traces/data_audio.csv"]
    network_trace = "datasets/network_traces/trace.txt"
    
    # The file path of packets' log
    log_packet_file = "output/packet_log/packet-0.log"
    # Select the solution file
    solution_file = 'solution_demos.reno.solution'

    evaluate(solution_file, block_traces, network_trace, log_packet_file)


