# Rust Code Documentation - People Detection with OpenCV and MQTT

## Overview
This Rust program captures video from a webcam, detects people using OpenCV's HOGDescriptor, and publishes the count of detected people to an MQTT broker. It also displays the video feed with detection boxes in a window.

## Dependencies

The program utilizes the following Rust crates:
- `opencv` - For image processing and object detection.
- `tokio` - For asynchronous programming.
- `rumqttc` - For MQTT communication.
- `chrono` - For timestamp generation.

## Code Breakdown

### Imports
```rust
use opencv::{
    core::{self, Size, Mat},
    highgui, imgproc, prelude::*, videoio,
    objdetect::HOGDescriptor,
    types::VectorOfRect,
};
use std::env;
use std::sync::Arc;
use tokio::task;
use tokio::time::{self, Duration};
use rumqttc::{MqttOptions, AsyncClient, QoS};
use chrono::{Local};
```
- The `opencv` crate is used for image processing and object detection.
- `tokio` provides async support.
- `rumqttc` allows communication with an MQTT broker.
- `chrono` helps with timestamps.

### Timestamp Function
```rust
fn get_timestamp() -> String {
    let now = Local::now();
    now.format("[%Y/%m/%d/%H/%M/%S%.3f]").to_string() // Add milliseconds
}
```
- Generates a timestamp in the format `[YYYY/MM/DD/HH/MM/SS.mmm]`.

### Colored Log Function
```rust
fn colored_log(message: &str, color_code: &str) -> String {
    format!("{}{}{}", color_code, message, "\x1b[0m")
}
```
- Adds ANSI color codes to log messages for better terminal readability.

### Main Function
```rust
#[tokio::main]
async fn main() -> opencv::Result<()> {
```
- Uses `#[tokio::main]` to run the asynchronous function as the main entry point.

### Default Configuration
```rust
let default_camera_index = 2;
let default_camera_frame_width = 1280.0;
let default_camera_frame_height = 720.0;
let default_broker_ip = "192.168.1.78".to_string();
let default_broker_ip_port = 1883;
```
- Sets default values for the camera and MQTT broker.

### Parsing Command-line Arguments
```rust
let args: Vec<String> = env::args().collect();
let camera_index = if args.len() > 1 { args[1].parse().unwrap_or(default_camera_index) } else { default_camera_index };
let camera_frame_width = if args.len() > 2 { args[2].parse().unwrap_or(default_camera_frame_width) } else { default_camera_frame_width };
let camera_frame_height = if args.len() > 3 { args[3].parse().unwrap_or(default_camera_frame_height) } else { default_camera_frame_height };
let broker_ip = if args.len() > 4 { args[4].clone() } else { default_broker_ip.clone() };
let broker_ip_port = if args.len() > 5 { args[5].parse().unwrap_or(default_broker_ip_port) } else { default_broker_ip_port };
```
- Reads user-provided arguments for camera index, resolution, and MQTT settings.

### Initializing MQTT Client
```rust
let mut mqttoptions = MqttOptions::new("person_detector", broker_ip, broker_ip_port);
mqttoptions.set_keep_alive(Duration::from_secs(60));
let (client, mut eventloop) = AsyncClient::new(mqttoptions, 10);
let client = Arc::new(client);
```
- Creates an MQTT client that connects to the given broker.

### Initializing HOG Descriptor
```rust
let mut hog = HOGDescriptor::default()?;
hog.set_svm_detector(&HOGDescriptor::get_default_people_detector()?)?;
```
- Initializes OpenCV’s HOG descriptor with a pre-trained people detector.

### Opening the Camera
```rust
let mut cam = videoio::VideoCapture::new(camera_index, videoio::CAP_ANY)?;
if !cam.is_opened()? {
    panic!("Unable to open default camera!");
}
```
- Opens the specified camera and verifies if it is accessible.

### Setting Camera Resolution
```rust
cam.set(videoio::CAP_PROP_FRAME_WIDTH, camera_frame_width)?;
cam.set(videoio::CAP_PROP_FRAME_HEIGHT, camera_frame_height)?;
```
- Configures the camera’s frame width and height.

### Creating a Display Window
```rust
highgui::named_window("People Detection", highgui::WINDOW_AUTOSIZE)?;
```
- Creates an OpenCV window to display the processed frames.

### Processing Video Frames
```rust
loop {
    let mut frame = Mat::default();
    cam.read(&mut frame)?;
    if frame.empty() {
        time::sleep(Duration::from_millis(1)).await;
        continue;
    }
```
- Captures frames from the camera.
- Skips empty frames.

### Converting to Grayscale
```rust
let mut processed_frame = Mat::default();
imgproc::cvt_color(&frame, &mut processed_frame, imgproc::COLOR_BGR2GRAY, 0)?;
```
- Converts the frame to grayscale for better detection performance.

### Detecting People
```rust
let mut boxes = VectorOfRect::new();
hog.detect_multi_scale(
    &processed_frame,
    &mut boxes,
    0.88,
    Size::new(8, 8),
    Size::new(26, 26),
    1.03,
    2.0,
    false,
)?;
```
- Uses OpenCV’s `detect_multi_scale()` to detect people in the frame.
- The function parameters control detection sensitivity.

### Counting and Logging Detections
```rust
let people_count = boxes.len();
let timestamp = get_timestamp();
let message = format!(
    "{} - {}People Detected: {}",
    colored_log(&timestamp, "\x1b[33m"),
    colored_log("People Detected: ", "\x1b[37m"),
    colored_log(&people_count.to_string(), "\x1b[32m")
);
println!("{}", message);
```
- Logs the number of detected people with timestamp and color formatting.

### Sending MQTT Message
```rust
let send_message = format!("{}", &people_count.to_string());
let client = Arc::clone(&client);
task::spawn(async move {
    if let Err(e) = client.publish("person_detector", QoS::AtLeastOnce, false, send_message).await {
        eprintln!("Failed to publish message: {}", e);
    }
});
```
- Publishes the detected people count to the MQTT broker.

### Drawing Detection Boxes
```rust
for rect in boxes.iter() {
    imgproc::rectangle(&mut frame, rect, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_AA, 0)?;
}
```
- Draws green rectangles around detected people.

### Displaying Processed Frames
```rust
highgui::imshow("People Detection", &frame)?;
if highgui::wait_key(1)? == 'q' as i32 { break; }
```
- Displays the processed frames and allows exiting with the ‘q’ key.

### Cleaning Up
```rust
cam.release()?;
highgui::destroy_all_windows()?;
```
- Releases the camera and closes the window upon exit.

## Conclusion
This program effectively detects people in real-time using OpenCV and publishes the count via MQTT. Let me know if you need modifications!
