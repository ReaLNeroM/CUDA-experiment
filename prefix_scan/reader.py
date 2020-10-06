import sys
import re
filename = sys.argv[1]

total_time_taken_seconds = 0.0
total_attempts = 0

with open(filename, 'r') as file:
    for line in file:
        if "memory usage" in line:
            memory_usage = int(line.split(' ')[2])
        elif "scan(" in line or "prefix_scan(" in line:
            time_taken_string = re.findall("[\\d\.]+\w+", line)[1]
            # print(time_taken_string)

            time_taken_float = float(time_taken_string[:-2])
            total_attempts += 1
            if time_taken_string[-2:] == "ms":
                total_time_taken_seconds += time_taken_float / 1e3
            elif time_taken_string[-2:] == "us":
                total_time_taken_seconds += time_taken_float / 1e6
        elif "propagate(" in line:
            time_taken_string = re.findall("[\\d\.]+\w+", line)[1]

            time_taken_float = float(time_taken_string[:-2])
            if time_taken_string[-2:] == "ms":
                total_time_taken_seconds += time_taken_float / 1e3
            elif time_taken_string[-2:] == "us":
                total_time_taken_seconds += time_taken_float / 1e6
        elif "prefix_scan: " in line:
            time_taken_string = re.findall("[\\d\.]+\w+", line)[0]

            time_taken_float = float(time_taken_string[:-2])
            total_attempts += 1
            if time_taken_string[-2:] == "ms":
                total_time_taken_seconds += time_taken_float / 1e3
            elif time_taken_string[-2:] == "us":
                total_time_taken_seconds += time_taken_float / 1e6


avg_time_taken = total_time_taken_seconds / total_attempts
bandwidth = memory_usage / avg_time_taken
print("{:.3f} ms, {:.2f} GBPS".format(avg_time_taken * 1e3, bandwidth / 1e9))
