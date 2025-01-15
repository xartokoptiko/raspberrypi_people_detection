use opencv::{
    core::{self, Size, Mat},
    highgui, imgproc, prelude::*, videoio,
    objdetect::HOGDescriptor,
    types::VectorOfRect,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task;
use tokio::time::{self, Duration};
use rumqttc::{MqttOptions, AsyncClient, QoS};

#[derive(Clone, Debug)]
struct Person {
    id: usize,
    confidence_percentage: f32,
    bbox: core::Rect,
}

impl PartialEq for Person {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && (self.bbox & other.bbox).area() > 0
    }
}

fn non_maximum_suppression(boxes: &VectorOfRect, threshold: f32) -> VectorOfRect {
    let mut results = VectorOfRect::new();

    for i in 0..boxes.len() {
        let rect1 = boxes.get(i).unwrap();
        let mut should_keep = true;

        for j in 0..boxes.len() {
            if i == j {
                continue;
            }

            let rect2 = boxes.get(j).unwrap();
            let intersection = rect1 & rect2;
            let union = rect1.area() + rect2.area() - intersection.area();

            if intersection.area() as f32 / union as f32 > threshold {
                should_keep = false;
                break;
            }
        }

        if should_keep {
            results.push(rect1);
        }
    }

    results
}

#[tokio::main]
async fn main() -> opencv::Result<()> {
    // Initialize MQTT client
    let mut mqttoptions = MqttOptions::new("person_detector", "---.---.-.-", 1883);
    mqttoptions.set_keep_alive(Duration::from_secs(60));
    let (client, mut eventloop) = AsyncClient::new(mqttoptions, 10);
    let client = Arc::new(client);

    // Initialize the HOG descriptor with optimized parameters
    let mut hog = HOGDescriptor::default()?;
    hog.set_svm_detector(&HOGDescriptor::get_default_people_detector()?)?;

    // Open webcam video stream
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("Unable to open default camera!");
    }

    // Increase resolution for better detection
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;

    highgui::named_window("Frame", highgui::WINDOW_AUTOSIZE)?;

    let prev_people_map: Arc<Mutex<HashMap<usize, Person>>> = Arc::new(Mutex::new(HashMap::new()));

    let mut next_person_id = 1; // Counter to assign unique IDs to each person

    // Spawn a task to handle MQTT event loop
    let mqtt_client = Arc::clone(&client);
    tokio::spawn(async move {
        loop {
            match eventloop.poll().await {
                Ok(_) => println!("MQTT event loop running."),
                Err(e) => eprintln!("MQTT error: {}", e),
            }
        }
    });

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;

        if frame.empty() {
            time::sleep(Duration::from_millis(1)).await;
            continue;
        }

        // Preprocessing steps for better detection
        let mut processed_frame = Mat::default();
        imgproc::gaussian_blur(
            &frame,
            &mut processed_frame,
            Size::new(3, 3),
            0.0,
            0.0,
            core::BORDER_DEFAULT,
        )?;
        let mut gray = Mat::default();
        imgproc::cvt_color(&processed_frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Detect people with optimized parameters
        let mut boxes = VectorOfRect::new();
        hog.detect_multi_scale(
            &processed_frame,
            &mut boxes,
            0.2,
            Size::new(8, 8),
            Size::new(32, 32),
            1.05,
            5.0,
            false,
        )?;

        // Apply non-maximum suppression
        let filtered_boxes = non_maximum_suppression(&boxes, 0.5);

        let mut new_people_map: HashMap<usize, Person> = HashMap::new();
        let mut used_ids = vec![];
        let mut has_changed = false;

        {
            let mut prev_map = prev_people_map.lock().await;

            if filtered_boxes.is_empty() {
                if !prev_map.is_empty() {
                    has_changed = true; // Only trigger change detection if previously there were people
                    prev_map.clear();   // Clear tracking when no one is in the frame
                    next_person_id = 1; // Reset the person ID counter
                }
            } else {
                for rect in filtered_boxes.iter() {
                    // Compute confidence based on bounding box area
                    let confidence = (rect.area() as f32) / 1000.0; // Simple heuristic
                    let confidence_percentage = confidence.min(100.0); // Clamp to 100%

                    if rect.width > 60 && rect.height > 120 {
                        // Try to match with previous detections
                        let mut matched_id = None;

                        for (&id, person) in prev_map.iter() {
                            if (person.bbox & rect).area() > 0 {
                                matched_id = Some(id);
                                break;
                            }
                        }

                        let id = matched_id.unwrap_or_else(|| {
                            let new_id = next_person_id;
                            next_person_id += 1;
                            new_id
                        });

                        used_ids.push(id);
                        new_people_map.insert(
                            id,
                            Person {
                                id,
                                confidence_percentage,
                                bbox: rect.clone(),
                            },
                        );
                    }
                }

                // Check for changes in detection results
                if *prev_map != new_people_map {
                    has_changed = true;
                }

                // Update prev_map with the new people map
                *prev_map = new_people_map.clone();
            }
        }

        // Only publish or print if something has changed
        if has_changed {
            if !new_people_map.is_empty() {
                let client = Arc::clone(&client);
                let message = new_people_map
                    .iter()
                    .map(|(id, person)| format!("{}: Person {} - {:.2}%", id, person.id, person.confidence_percentage))
                    .collect::<Vec<String>>()
                    .join("\n");

                println!("Detected persons:\n{}", message);

                task::spawn(async move {
                    println!("Attempting to publish to MQTT...");
                    if let Err(e) = client.publish("person_detector", QoS::AtLeastOnce, false, message).await {
                        eprintln!("Failed to publish message: {}", e);
                    } else {
                        println!("Message successfully published to MQTT.");
                    }
                });
            } else {
                println!("No people detected.");
            }
        }

        // Draw bounding boxes
        let mut processed_frame = frame.clone();
        for person in new_people_map.values() {
            imgproc::rectangle(
                &mut processed_frame,
                person.bbox,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;

            let label = format!("Person {} - {:.2}%", person.id, person.confidence_percentage);
            imgproc::put_text(
                &mut processed_frame,
                &label,
                core::Point::new(person.bbox.x, person.bbox.y - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }

        highgui::imshow("Frame", &processed_frame)?;

        if highgui::wait_key(1)? == 'q' as i32 {
            break;
        }
    }

    cam.release()?;
    highgui::destroy_all_windows()?;

    Ok(())

}



