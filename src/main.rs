use opencv::{
    core::{self, Rect, Size, Mat},
    highgui, imgproc, prelude::*, videoio,
    objdetect::HOGDescriptor,
    types::{VectorOfRect, VectorOff32},
};
use std::time::Duration;

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

fn main() -> opencv::Result<()> {
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

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;

        if frame.empty() {
            std::thread::sleep(Duration::from_millis(1));
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
        let mut thresholded = Mat::default();
        imgproc::adaptive_threshold(&gray, &mut thresholded, 255.0, imgproc::ADAPTIVE_THRESH_MEAN_C, imgproc::THRESH_BINARY, 11, 2.0)?;

        // Detect people with optimized parameters
        let mut boxes = VectorOfRect::new();
        hog.detect_multi_scale(
            &processed_frame,
            &mut boxes,
            0.2,                // hit_threshold
            Size::new(8, 8),    // win_stride
            Size::new(32, 32),  // padding
            1.05,               // scale
            5.0,                // final_threshold
            false,              // use_meanshift_grouping
        )?;

        // Apply non-maximum suppression
        let filtered_boxes = non_maximum_suppression(&boxes, 0.5);

        // Filter and draw detections
        for rect in filtered_boxes.iter() {
            // Compute confidence based on bounding box area
            let confidence = (rect.area() as f32) / 1000.0; // Simple heuristic, you can modify this logic
            let confidence_percentage = confidence.min(100.0); // Clamp to 100%

            if rect.width > 60 && rect.height > 120 {
                imgproc::rectangle(
                    &mut processed_frame,
                    rect,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_AA,
                    0,
                )?;

                // Display dynamic confidence score based on box area
                let label = format!("Person: {:.2}%", confidence_percentage);
                let font_scale = 0.5;
                let thickness = 1;
                let font = imgproc::FONT_HERSHEY_SIMPLEX;
                let mut baseline = 0;
                imgproc::get_text_size(&label, font, font_scale, thickness, &mut baseline)?;

                imgproc::put_text(
                    &mut processed_frame,
                    &label,
                    core::Point::new(rect.x, rect.y - 10),
                    font,
                    font_scale,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    thickness,
                    imgproc::LINE_AA,
                    false,
                )?;
            }
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
