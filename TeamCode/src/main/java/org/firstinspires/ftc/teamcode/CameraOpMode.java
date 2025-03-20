package org.firstinspires.ftc.teamcode;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.robotcore.hardware.Servo;
import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;

import org.openftc.easyopencv.OpenCvCamera;
import org.openftc.easyopencv.OpenCvCameraFactory;
import org.openftc.easyopencv.OpenCvCameraRotation;
import org.openftc.easyopencv.OpenCvPipeline;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

@TeleOp(name = "Camera Servo Control OpMode", group = "Linear Opmode")
public class CameraOpMode extends LinearOpMode {

    private Servo spinServo;
    private Servo upDownServo;
    private OpenCvCamera cvCamera;

    private Mat hsvMat = new Mat();
    private Mat thresholdMat = new Mat();

    // Yellow color range in HSV
    private static final Scalar YELLOW_LOWER = new Scalar(20, 100, 100);
    private static final Scalar YELLOW_UPPER = new Scalar(40, 255, 255);

    private boolean yellowBlockDetected = false;
    private double detectedBlockX = -1;
    private double detectedBlockY = -1;

    @Override
    public void runOpMode() {
        // Initialize servos
        spinServo = hardwareMap.get(Servo.class, "spin_servo");
        upDownServo = hardwareMap.get(Servo.class, "up_down_servo");

        // Setup the camera
        int cameraMonitorViewId = hardwareMap.appContext.getResources().getIdentifier(
                "cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());

        cvCamera = OpenCvCameraFactory.getInstance().createWebcam(
                hardwareMap.get(WebcamName.class, "Webcam 1"), cameraMonitorViewId);

        cvCamera.setPipeline(new YellowBlockDetectionPipeline());

        // Open camera asynchronously
        cvCamera.openCameraDeviceAsync(new OpenCvCamera.AsyncCameraOpenListener() {
            @Override
            public void onOpened() {
                cvCamera.startStreaming(640, 480, OpenCvCameraRotation.UPRIGHT);
            }

            @Override
            public void onError(int errorCode) {
                telemetry.addData("Camera Error", errorCode);
                telemetry.update();
            }
        });

        waitForStart();

        while (opModeIsActive()) {

            if (yellowBlockDetected) {

                double spinPosition = mapToServoRange(detectedBlockX, 0, 640, 0, 1);
                spinServo.setPosition(spinPosition);

                double upDownPosition = mapToServoRange(detectedBlockY, 0, 480, 0, 1);
                upDownServo.setPosition(upDownPosition);
            }

            telemetry.addData("Yellow Block Detected", yellowBlockDetected);
            telemetry.addData("Block X", detectedBlockX);
            telemetry.addData("Block Y", detectedBlockY);
            telemetry.update();
        }

        hsvMat.release();
        thresholdMat.release();
    }

    private double mapToServoRange(double value, double minInput, double maxInput, double minOutput, double maxOutput) {
        return (value - minInput) / (maxInput - minInput) * (maxOutput - minOutput) + minOutput;
    }

    private class YellowBlockDetectionPipeline extends OpenCvPipeline {

        @Override
        public Mat processFrame(Mat input) {

            Imgproc.cvtColor(input, hsvMat, Imgproc.COLOR_RGB2HSV);

            Core.inRange(hsvMat, YELLOW_LOWER, YELLOW_UPPER, thresholdMat);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(thresholdMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            if (contours.size() > 0) {
                double maxArea = 0;
                int maxAreaIndex = -1;
                for (int i = 0; i < contours.size(); i++) {
                    double area = Imgproc.contourArea(contours.get(i));
                    if (area > maxArea) {
                        maxArea = area;
                        maxAreaIndex = i;
                    }
                }

                if (maxAreaIndex >= 0) {
                    // Get the bounding box for the largest contour
                    Rect boundingRect = Imgproc.boundingRect(contours.get(maxAreaIndex));

                    // Calculate the center of the detected block
                    detectedBlockX = boundingRect.x + boundingRect.width / 2.0;
                    detectedBlockY = boundingRect.y + boundingRect.height / 2.0;

                    // Mark the contour with a rectangle on the frame
                    Imgproc.rectangle(input, boundingRect.tl(), boundingRect.br(), new Scalar(0, 255, 0), 2);
                    yellowBlockDetected = true;
                }
            } else {
                // No contour found, reset values
                yellowBlockDetected = false;
                detectedBlockX = -1;
                detectedBlockY = -1;
            }

            hierarchy.release();
            return input;
        }
    }
}
