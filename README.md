# DUET: Improving Inertial-based Odometry via Deep IMU Online Calibration

This repository contains the code to [our paper](https://ieeexplore.ieee.org/document/10225410).
For questions feel free to open an issue or send an e-mail to <liu.huakun.li0@is.naist.jp>.

## Getting started

> This code was tested on Arch Linux with Python 3.10, PyTorch 2.2 and CUDA 12.3.

1. Clone the repository onto your local system.
2. Create a virtual environment and activate the created virtual environment(tested on Python 3.10).
3. Install the necessary packages from the requirements file with:

   ```shell
   python -m pip install -r requirements.txt
   ```

4. Follow `main_EuRoC.py` to build your own train test flow.
5. (Optinal) Download the dataset (_EuRoC [1]_ and _TUM-VI [2]_) and decompress it in `data` folder.

## Citation

If you find the project helpful, or use the code or paper from this repository in your research, please consider citing us:

```
@article{liu2023duet,
  author={H. {Liu} and X. {Wei} and M. {Perusquía-Hernández} and I. {Naoya} and H. {Uchiyama} and K. {Kiyokawa}},
  journal={IEEE Transactions on Instrumentation and Measurement},
  title={DUET: Improving Inertial-Based Odometry via Deep IMU Online Calibration},
  year={2023},
  volume={72},
  number={},
  pages={1-13},
}
```

---

[1] M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. W. Achtelik, and R. Siegwart, ``The EuRoC Micro Aerial Vehicle Datasets", The International Journal of Robotics Research, vol. 35, no. 10, pp. 1157–1163, 2016.

[2] D. Schubert, T. Goll, N. Demmel, V. Usenko, J. Stuckler, and D. Cremers, ``The TUM VI Benchmark for Evaluating Visual-Inertial Odometry", in International Conference on Intelligent Robots and Systems (IROS). IEEE, pp. 1680–1687, 2018.
