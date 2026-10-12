# Belief Propagation for Stereo Matching

## Overview

This project implements the **Belief Propagation (BP)** algorithm for stereo matching using C++ and OpenCV.

Given a pair of stereo images, the program uses the left image as the reference to estimate a disparity map. The resulting disparity map is displayed and saved to disk. During execution, intermediate disparity maps and their corresponding energy values are displayed to illustrate the algorithm's progress.

The implementation supports configurable algorithm parameters, six message update schedules, alternative message computation methods, optional filtering, and standard and low-memory execution modes. It also includes motion-based disparity estimation for processing consecutive stereo image pairs.

## Features

- **Configurable Parameters:** Customize the algorithm through a YAML configuration file (`Options.yml`) without recompiling the program.
- **Multiple Message Update Schedules:** Choose from six message update schedules.
- **Energy Evaluation:** Calculate and display the energy associated with the disparity map at each iteration.
- **Window-Based Data Cost:** Optionally use a configurable window when computing the data cost.
- **Gaussian Filtering:** Enable or disable Gaussian filtering.
- **Alternative Message Computation Methods:** Choose between standard BP message computation and an SGM-inspired method.
- **Standard and Low-Memory Modes:** Select between the standard execution mode and an alternative mode designed to reduce memory requirements.
- **Colorized Disparity Maps:** Optionally generate a colorized representation of the disparity map.
- **Motion-Based Disparity Estimation:** Use disparity labels from previous results to influence the estimation of subsequent stereo image pairs.

## Technologies

- **Programming Language:** C++
- **IDE:** Microsoft Visual Studio 2019
- **Computer Vision Library:** OpenCV 4.10.0
- **Configuration Format:** YAML

## Setup

1. Install Microsoft Visual Studio 2019.
2. Download and configure OpenCV 4.10.0 for Visual Studio.
3. Set the `OPENCV_DIR` environment variable to the OpenCV `vc16` directory. For example:

```
   OPENCV_DIR=C:\opencv\build\x64\vc16
```

4. Open the project or solution in Visual Studio.
5. Ensure that the OpenCV configuration matches the selected build configuration and platform.
6. Build the solution.

## Input Data

In standard mode, the program expects a directory containing a stereo image pair with identical image dimensions:

- `Left.png` — The left reference image.
- `Right.png` — The corresponding right image.

The input directory is specified by the `Input Folder` parameter in `Options.yml`.

Example directory structure:

```
Data/
└── Tsukuba/
    ├── Left.png
    └── Right.png
```

When motion-based processing is enabled, the program uses a sequence of five stereo image pairs stored in the `Motion/` subdirectory. See Motion-Based Disparity Estimation for details.

## Configuration

The program loads its configuration from `Options.yml` at startup. This file controls the disparity search space, message update schedule, number of iterations, filtering, message computation method, memory usage, output visualization, and motion-related settings.

### Example Configuration

```
%YAML:1.0
Input Folder: "./Data/Tsukuba/"
Disparity Levels: 16
Update Schedule: 2
Iterations: 40
Gaussian Filter: 1
Window Size: 1
SGM Like: 0
Smoothness Weight: 5
Smoothness Threshold: 2
SGM P1: 5
SGM P2: 10
Low Memory: 0
Color Map: 0
Motion: 0
Previous Labels Cost: 0
```

### Parameter Reference

| Parameter | Description | Supported values |
| --- | --- | --- |
| `Input Folder` | Path to the directory containing `Left.png` and `Right.png` for standard processing. | A valid directory path containing the required images. |
| `Disparity Levels` | Number of disparity levels considered during stereo matching. | Positive integer; implementation-specific limits apply. |
| `Update Schedule` | Selects the message update schedule used by the BP algorithm. | Integers from `1` to `6`; see the schedule reference below. |
| `Iterations` | Number of iterations performed by the algorithm. | Positive integer; implementation-specific limits apply. |
| `Gaussian Filter` | Controls whether Gaussian filtering is applied. | `0` = Disabled, `1` = Enabled. |
| `Window Size` | Controls the window size used for window-based data cost computation. | `1` avoids using a larger window; other valid values depend on the implementation. |
| `SGM Like` | Selects the message computation method. | `0` = Standard BP, `1` = SGM-inspired method. |
| `Smoothness Weight` | Weight ($\lambda$) controlling the contribution of the smoothness cost in standard BP message computation. | Numeric value; implementation-specific limits apply. |
| `Smoothness Threshold` | Threshold ($T$) used by the smoothness cost function in standard BP message computation. | Numeric value; implementation-specific limits apply. |
| `SGM P1` | Penalty $P_1$ used by the SGM-inspired message computation method. | Numeric value; implementation-specific limits apply. |
| `SGM P2` | Penalty $P_2$ used by the SGM-inspired message computation method. | Numeric value; implementation-specific limits apply. |
| `Low Memory` | Selects the execution mode. When enabled, only the `ByRow1` and `ByRow2` message update schedules are supported. | `0` = Standard mode, `1` = Low-memory mode. |
| `Color Map` | Controls whether the output disparity map is colorized. | `0` = Disabled, `1` = Enabled. |
| `Motion` | Enables or disables motion-based processing using previous disparity results. | `0` = Disabled, `1` = Enabled. |
| `Previous Labels Cost` | Initialization cost assigned to disparity labels used in the previous disparity map when motion-based processing is enabled. All other disparity labels are initialized with a cost of `0`. | Negative integer values are recommended to encourage reuse of previous labels. |

### Message Update Schedules

The `Update Schedule` parameter selects one of six message update schedules.

| Value | Method |
| --- | --- |
| `1` | Synchronous |
| `2` | Bipartite |
| `3` | BP-M (Accelerated) |
| `4` | BP-S |
| `5` | ByRow1 |
| `6` | ByRow2 |

When `Low Memory = 1`, only schedules `5` and `6` are supported.

### Standard BP and SGM-Inspired Message Computation

The program supports two alternative methods for computing messages during belief propagation: standard BP and an SGM-inspired method. The `SGM Like` parameter selects the method.

#### Standard BP

When `SGM Like = 0`, the program uses the standard BP message computation method, which relies on a smoothness cost function. The `Smoothness Weight` and `Smoothness Threshold` parameters control the contribution and behavior of the smoothness cost.

#### SGM-Inspired Message Computation

When `SGM Like = 1`, the program uses an alternative message computation method inspired by Semi-Global Matching (SGM). This method uses the penalty terms $P_1$ and $P_2$, configured through `SGM P1` and `SGM P2`, respectively, instead of the standard BP smoothness cost function.

#### Configuration Examples

To use standard BP:

```
SGM Like: 0
Smoothness Weight: 5
Smoothness Threshold: 2
```

To use the SGM-inspired method:

```
SGM Like: 1
SGM P1: 5
SGM P2: 10
```

The message computation method can be selected independently of the message update schedule, subject to the restrictions of the selected execution mode.

### Standard and Low-Memory Modes

The program provides two execution modes, controlled by the `Low Memory` parameter.

#### Standard Mode

When `Low Memory = 0`, the program uses the standard execution mode, in which all six message update schedules are supported.

#### Low-Memory Mode

When `Low Memory = 1`, the program uses an alternative execution mode designed to reduce memory requirements. Only the `ByRow1` and `ByRow2` message update schedules are supported in this mode.

#### Configuration Examples

To use standard mode:

```
Low Memory: 0
Update Schedule: 2
```

To use low-memory mode:

```
Low Memory: 1
Update Schedule: 5
```

When low-memory mode is enabled, set `Update Schedule` to either `5` or `6`.

### Motion-Based Disparity Estimation

When `Motion = 1`, the program processes a sequence of five consecutive stereo image pairs stored in the `Motion/` subdirectory. The sequence consists of the following ten images:

- `Left_1.png` through `Left_5.png` — The left images.
- `Right_1.png` through `Right_5.png` — The corresponding right images.

The program uses the previous disparity map to influence the estimation of the current disparity map. This functionality is intended for consecutive stereo image pairs, such as frames from a stereo video sequence.

#### Previous Labels Cost

The `Previous Labels Cost` parameter controls how disparity labels from the previous result influence message initialization:

- Disparity labels present in the previous disparity map are initialized with the specified cost.
- All other disparity labels are initialized with a cost of `0`.
- **Negative values** encourage the algorithm to reuse disparity labels from the previous result.
- Zero or positive values are allowed, but they do not provide the intended incentive to reuse previous labels.

For example, setting `Previous Labels Cost` to `-5` assigns an initial cost of `-5` to disparity labels present in the previous disparity map, while all other labels receive `0`.

This mechanism influences message initialization but does not force the algorithm to select the previous disparity labels in the final result.

## Configuration Notes

- Ensure that `Options.yml` contains valid YAML syntax and parameter values supported by the implementation.
- Boolean options use `0` to disable a feature and `1` to enable it.
- Setting `Window Size` to `1` avoids using a larger window for data cost computation.
- `Smoothness Weight` and `Smoothness Threshold` are relevant to standard BP message computation.
- `SGM P1` and `SGM P2` are relevant to the SGM-inspired message computation method.
- When `Low Memory = 1`, use `Update Schedule = 5` or `Update Schedule = 6`.
- `Previous Labels Cost` influences message initialization when motion-based processing is enabled.
- Invalid parameter values may cause unexpected behavior because the program expects valid configuration values.

## About
This repository contains the source code developed for a master's thesis. You can find the full thesis text [here](https://repository.ihu.gr/handle/11544/49252).

## References

- _D. Scharstein and R. Szeliski. [A taxonomy and evaluation of dense two-frame stereo correspondence algorithms](https://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf). International Journal of Computer Vision, 47(1/2/3):7-42, April-June 2002._
- _D. Scharstein and R. Szeliski. [High-accuracy stereo depth maps using structured light](http://www.cs.middlebury.edu/~schar/papers/structlight/). In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2003), volume 1, pages 195-202, Madison, WI, June 2003._
- _D. Scharstein and C. Pal. [Learning conditional random fields for stereo](http://www.cs.middlebury.edu/~schar/papers/LearnCRFstereo_cvpr07.pdf). In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2007), Minneapolis, MN, June 2007._
- _H. Hirschmüller and D. Scharstein. [Evaluation of cost functions for stereo matching](http://www.cs.middlebury.edu/~schar/papers/evalCosts_cvpr07.pdf). In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2007), Minneapolis, MN, June 2007._

## License

The source code, scripts, and other original materials in this repository are licensed under the Creative Commons [Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/) (CC BY-NC-ND 4.0) license.

The `Data/` directory contains images derived from the [Middlebury Stereo Datasets](<https://vision.middlebury.edu/stereo/data/>), including modified versions of the original images. These materials are subject to the applicable terms of their original source. Please cite the relevant Middlebury publications when using the datasets.

