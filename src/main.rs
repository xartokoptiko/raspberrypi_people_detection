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

fn get_timestamp() -> String {
    let now = Local::now();
    now.format("[%Y/%m/%d/%H/%M/%S%.3f]").to_string()  // Add milliseconds
}

fn colored_log(message: &str, color_code: &str) -> String {
    format!("{}{}{}", color_code, message, "\x1b[0m")
}

#[tokio::main]
async fn main() -> opencv::Result<()> {
    // Default values
    let default_camera_index = 2;
    let default_camera_frame_width = 1280.0;
    let default_camera_frame_height = 720.0;
    let default_broker_ip = "192.168.1.78".to_string();
    let default_broker_ip_port = 1883;

    // Collect command-line arguments
    let args: Vec<String> = env::args().collect();
    let camera_index = if args.len() > 1 {
        args[1].parse().unwrap_or(default_camera_index)
    } else {
        default_camera_index
    };

    let camera_frame_width = if args.len() > 2 {
        args[2].parse().unwrap_or(default_camera_frame_width)
    } else {
        default_camera_frame_width
    };

    let camera_frame_height = if args.len() > 3 {
        args[3].parse().unwrap_or(default_camera_frame_height)
    } else {
        default_camera_frame_height
    };

    let broker_ip = if args.len() > 4 {
        args[4].clone()
    } else {
        default_broker_ip.clone()
    };

    let broker_ip_port = if args.len() > 5 {
        args[5].parse().unwrap_or(default_broker_ip_port)
    } else {
        default_broker_ip_port
    };

    // Initialize MQTT client
    let mut mqttoptions = MqttOptions::new("person_detector", broker_ip, broker_ip_port);
    mqttoptions.set_keep_alive(Duration::from_secs(60));
    let (client, mut eventloop) = AsyncClient::new(mqttoptions, 10);
    let client = Arc::new(client);

    // Initialize the HOG descriptor
    let mut hog = HOGDescriptor::default()?;
    hog.set_svm_detector(&HOGDescriptor::get_default_people_detector()?)?;

    // Open webcam video stream
    let mut cam = videoio::VideoCapture::new(camera_index, videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("Unable to open default camera!");
    }

    // Set camera resolution
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, camera_frame_width)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, camera_frame_height)?;

    highgui::named_window("People Detection", highgui::WINDOW_AUTOSIZE)?;

    loop {

        let mut frame = Mat::default();
        cam.read(&mut frame)?;

        if frame.empty() {
            time::sleep(Duration::from_millis(1)).await;
            continue;
        }

        let mut processed_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut processed_frame, imgproc::COLOR_BGR2GRAY, 0)?;

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

        let people_count = boxes.len();
        let timestamp = get_timestamp();
        let message = format!(
            "{} - {}People Detected: {}",
            colored_log(&timestamp, "\x1b[33m"),
            colored_log("People Detected: ", "\x1b[37m"),
            colored_log(&people_count.to_string(), "\x1b[32m")
        );

        let send_message = format!("{}", &people_count.to_string());


        println!("{}", message);

        let client = Arc::clone(&client);
        task::spawn(async move {
            if let Err(e) = client.publish("person_detector", QoS::AtLeastOnce, false, send_message).await {
                eprintln!("Failed to publish message: {}", e);
            }
        });

        // Draw detected people
        for rect in boxes.iter() {
            imgproc::rectangle(
                &mut frame,
                rect,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
        }

        highgui::imshow("People Detection", &frame)?;

        if highgui::wait_key(1)? == 'q' as i32 {
            break;
        }

        // Poll MQTT event loop
        eventloop.poll().await.expect("Failed to publish message to broker !");
    }

    cam.release()?;
    highgui::destroy_all_windows()?;

    Ok(())
}






