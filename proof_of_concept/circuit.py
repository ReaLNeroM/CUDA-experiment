# mypy: disallow-untyped-defs

"""
Implementation of .crc to CUDA converter using many copies of
the same array (# copies = # stages).
Assumes the .crc is the first argument.
"""

from typing import List, Set, Dict, Tuple
import sys
import re

ALLOWED_OPERATIONS = ["=", "+=", "-=", "*=", "/="]

"""
A line describes a circuit line. It stores a single data value, which may change between time units.
The data value is represented by the array name, and the index into that array.
"""
class Line:
    array_name: str
    index: int

    def __init__(self, array_name: str, index: int):
        self.array_name = array_name
        self.index = index
        pass

    def emit(self, time_step: int) -> str:
        """Emit the value of the data line at that particular time step."""
        return "{}{}[{}]".format(self.array_name, time_step, self.index)


"""
An operation describes a modification of a data line at a particular point in time.

Note that in this limited implementation, the only supported operations are
assignment variants such as =, +=, -=, *= etc.
"""
class Operation:
    time_step: int
    source_lines: List[Line]
    output_line: Line
    operation: str

    def __init__(
        self,
        time_step: int,
        source_lines: List[Line],
        output_line: Line,
        operation: str,
    ):
        self.time_step = time_step
        self.source_lines = source_lines
        self.output_line = output_line
        self.operation = operation

    def emit(self) -> str:
        """Emit the operation as a C++ line."""
        assert len(self.source_lines) == 1
        source_line: Line = self.source_lines[0]

        return "{} {} {};".format(
            self.output_line.emit(self.time_step + 1),
            self.operation,
            source_line.emit(self.time_step),
        )


"""
A circuit completely describes a GPU kernel.
In this limited implementation, kernel emitting works as follows:
    - Only int* arguments are supported.
    - Multiple copies of the array are made, to avoid data races due to concurrent accesses
      to the same data.
    - For each stage, each thread knows which operation to execute using a
      switch statement, which is dynamically generated using the operations the used has provided.
    - Between each stage, a __syncthreads() operation is inserted.
    - The output of the function will be stored at the last argument.
"""
class Circuit:
    name: str
    input_lines: List[Line]
    operations: List[List[Operation]]
    arg: str

    def __init__(
        self, name: str, input_lines_count: int, time_units_count: int, arg: str
    ):
        self.name = name
        self.input_lines = [Line(arg, i) for i in range(input_lines_count)]
        self.operations = [[] for _ in range(time_units_count)]
        self.arg = arg

    def fetch_line(self, i: int) -> Line:
        return self.input_lines[i]

    def fetch_lines(self, li: List[int]) -> List[Line]:
        return [self.fetch_line(i) for i in li]

    def add_operation(
        self,
        time: int,
        source_line_index: List[int],
        output_line_index: int,
        operation: str,
    ) -> None:
        op: Operation = Operation(
            time,
            self.fetch_lines(source_line_index),
            self.fetch_line(output_line_index),
            operation,
        )

        self.operations[time].append(op)

    def emit_lines(self) -> List[str]:
        """Emit the circuit as a CUDA C++ kernel."""
        kernel_lines: List[str] = []
        fn_args = [
            "int *{}{}".format(self.arg, i) for i in range(len(self.operations) + 1)
        ]
        # Kernel function declaration emission
        kernel_lines.append(
            "__global__ void {}({}){{".format(self.name, ", ".join(fn_args))
        )
        kernel_lines.append("int tid = threadIdx.x;")

        # Stage emission
        for ind, block in enumerate(self.operations):
            kernel_lines.append(
                "{}{}[tid] = {}{}[tid];".format(self.arg, ind + 1, self.arg, ind)
            )

            kernel_lines.append("switch(tid){")

            for operation in block:
                kernel_lines.append("case {}:".format(operation.output_line.index))
                # thread-specific operation emission
                kernel_lines.append(operation.emit())
                kernel_lines.append("break;")

            kernel_lines.append("}")

            # synchronization to ensure dependencies are propagated on time.
            if ind + 1 != len(self.operations):
                kernel_lines.append("__syncthreads();")

        kernel_lines.append("}")

        return kernel_lines

    def emit(self) -> str:
        """Emits the circuit as a well-indented C++ CUDA kernel."""
        kernel_lines = self.emit_lines()
        kernel_text = ""
        indentation = 0
        for ind, line in enumerate(kernel_lines):
            if "}" in line:
                indentation -= 1

            kernel_text += ("    " * indentation) + line + "\n"

            if "{" in line or ":" in line:
                indentation += 1
            elif "break;" in line:
                indentation -= 1

        return kernel_text


def read_circuit(file_name: str) -> Circuit:
    """Read a .crc format file, and return a Circuit."""
    with open(file_name, "r") as file:
        circuit: Circuit

        current_circuit_line = 0
        for index, line in enumerate(file):
            if line.isspace() or "#" in line:
                continue

            current_circuit_line += 1
            tokens: List[str] = list(filter(None, line.strip().split(" ")))
            if current_circuit_line == 1:
                # The first line has the format:
                #     circuit_name input_lines barrier_steps array_name
                tokens = line.strip().split(" ")
                assert len(tokens) == 4

                name = tokens[0]
                input_lines_count = int(tokens[1])
                time_units_count = int(tokens[2])
                arg = tokens[3]

                circuit = Circuit(name, input_lines_count, time_units_count, arg)
            else:
                # The other lines have the format:
                #     barrier_index result_line operation source_line
                tokens = list(filter(None, line.strip().split(" ")))
                assert len(tokens) == 4

                time: int = int(tokens[0])
                output_line: int = int(tokens[1])
                operation: str = tokens[2]
                source_line: int = int(tokens[3])

                assert operation in ALLOWED_OPERATIONS

                circuit.add_operation(time, [source_line], output_line, operation)

    return circuit


if __name__ == "__main__":
    file_name: str = sys.argv[1]
    circuit: Circuit = read_circuit(file_name)
    print(circuit.emit())
