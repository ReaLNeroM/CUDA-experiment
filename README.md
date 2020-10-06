# [Circuit to CUDA compiler](https://github.com/realnerom/Circuit-to-CUDA) &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/realnerom/Circuit-to-CUDA/blob/master/LICENSE)

Prototype of an alternative programming model for GPU-parallelized programs.

When programming in CUDA, we use an imperative language like C++ to describe our computation, and
leave it to the GPU to handle lower-level details like scheduling or multithreading. The research
thesis is that describing the computation in terms of circuits over the data is going to be a
worthwhile interface for prototyping GPU applications. These circuits would get compiled down to
native CUDA code using a compiler.

Slides are attached describing the project in high-level detail. For more details, consult the
Report PDF.

## 📚 Table of contents

- [Technical stack](#technical-stack)
- [Features](#features)
- [Running](#running)
- [Todo](#todo)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [License](#license)

## 🛠 Technical stack

- Programming language(s): C++, Python
- GPU Framework: [CUDA](https://developer.nvidia.com/cuda-zone)

## 🚀 Features

- Novel .crc format which can be used to describe computational circuits.
- A .crc to CUDA compiler.
- Optimized CUDA implementations of prefix scan algorithms.

## ⬇ Running

To run the CUDA optimizations benchmark, run

    cd prefix_scan
    ./test.sh

in a Unix-compatible terminal.

To run the CRC to CUDA compiler, run

    cd proof_of_concept
    python3 circuit2.py sum.crc
    python3 circuit2.py prefix_sum.crc

to get the CUDA kernel code.

## 📝 Todo

- Develop the idea for an inductive representation of a circuit.
- Add optimizations to the CUDA-generated code.

## 🎉 Acknowledgements

- professor Sreepathi Pai, for coming up with the idea, and assisting on CUDA concepts.

## 👨‍💻 Author

- [Vladimir Maksimovski](https://github.com/realnerom) <br/>
Bachelor of Science in Computer Science.
University of Rochester '22.

## 📄 License

The Circuit to CUDA compiler is [MIT licensed](./LICENSE).
