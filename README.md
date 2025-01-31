# Raspberry Pi People Detection

## Overview

This project utilizes a Raspberry Pi to detect people in real-time using a camera. It leverages OpenCV's HOG (Histogram of Oriented Gradients) object detection and MQTT for message publishing. The system detects people in the camera frame and sends the number of detected people to a specified MQTT broker.

## Features

- Real-time people detection using a webcam (Not a pyCam) connected to Raspberry Pi.
- MQTT-based communication to publish the count of detected people.
- Adjustable camera and broker parameters such as index, resolution, and broker configuration via command-line arguments.
- Timestamp logging with ANSI color formatting for better visualization in the terminal.

## Requirements

To run this project, you need:

- **Rust**: This project is built using Rust programming language. Ensure you have Rust version 1.82.0 or later installed.

  To install Rust, follow these steps:
    1. Install `rustup` by running:
        ```bash
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        ```
    2. Once `rustup` is installed, you can check the Rust version by running:
        ```bash
        rustc --version
        ```
    3. If you need to switch to version 1.82.0 (the required version), you can install it using:
        ```bash
        rustup install 1.82.0
        rustup default 1.82.0
        ```

- **Dependencies**: The exact dependencies required for this project will be listed here once identified. However, you may need the following to get started:
    - `opencv`: OpenCV bindings for Rust.
    - `rumqttc`: MQTT client for communication.

  You can add dependencies by including them in the `Cargo.toml` file. Example:

    ```toml
    [dependencies]
    opencv = "0.72"
    rumqttc = "0.13"
    tokio = { version = "1", features = ["full"] }
    chrono = "0.4"
    ```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/raspberry-pi-people-detection.git
    cd raspberry-pi-people-detection
    ```

2. Install the necessary dependencies:

    ```bash
    cargo build --release
    ```

3. Run the program:

    ```bash
    cargo run -- [camera_index] [frame_width] [frame_height] [broker_ip] [broker_port]
    ```

   Replace the placeholders with the appropriate values. If no arguments are provided, the program will use default values:
    - `camera_index`: 2
    - `frame_width`: 1280
    - `frame_height`: 720
    - `broker_ip`: 192.168.1.55
    - `broker_port`: 1883

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

