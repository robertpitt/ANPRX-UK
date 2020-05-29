package dev.robertpitt.anprX.opencv;

import androidx.camera.core.ImageProxy;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 *
 */
public class NumberplateDetectorV1 implements IDetector {
  /**
   * Log Tag
   */
  private static String TAG = "NumberplateDetector";

  /**
   * Working Input Frame
   */
  Mat frame;

  /**
   * Working HSV frame, used for color analysis
   */
  Mat frame_hsv;

  /**
   * Edges detected from frames.
   */
  private Mat frame_edges;

  /**
   *
   */
  private Mat front_mask_hsv;

  /**
   *
   */
  private Mat rear_mask_hsv;

  /**
   * HSV Threshold for front plate
   * H =
   * S = 0-30%
   * V = 10 % of 255
   */
  Scalar front_lower_range = new Scalar(0, 0, 210);
  Scalar front_upper_range = new Scalar(180, 220, 255);
  public Mat front_mask;

  /**
   * HSV Threshold for rear plate
   */
  Scalar rear_lower_range = new Scalar(20, 120, 60); //[20, 120, 60]
  Scalar rear_upper_range = new Scalar(50, 255, 255); //[60, 255, 255]
  Mat rear_mask;

  /**
   *
   */
  private boolean _initialised = false;

  /**
   * Initialise the Matrices
   */
  private void _initialise() {
    if(_initialised == false) {
      frame = new Mat();
      frame_hsv = new Mat();
      frame_edges = new Mat();
      front_mask = new Mat();
      rear_mask = new Mat();
      front_mask_hsv = new Mat();
      rear_mask_hsv = new Mat();
    } else {
      frame.release();
      frame_hsv.release();
      front_mask.release();
      rear_mask.release();
      frame_edges.release();
      front_mask_hsv.release();
      rear_mask_hsv.release();
    }
  }

  /**
   * Perform detection on an ImageProxy instance
   */
  public Mat detect(ImageProxy image) {
    return detect(Utils.imageToRGB(image));
  }

  /**
   * Process the working frame
   */
  public Mat detect(Mat src) {
    // Prep for the first frame and also do memory clear up between frames
    _initialise();

    // Resize the input to 640x480 into the frame pointer
    src.copyTo(frame);
//    Imgproc.resize(src, frame, new Size(1024, 480));

    // Create a HSV view of the image, this will be used for color filtering
    Imgproc.cvtColor(frame, frame_hsv, Imgproc.COLOR_RGB2HSV);

    // Create two masks, one for white front plates and one from yellow rear plates
    // this will filter out the pixels in the matrix that are not in the range given.
    // we have one for yellow and one for white, the masks output should be a
    // black 640x480 frame with white blocks in place where the color in range was detected.
    Core.inRange(frame_hsv, front_lower_range, front_upper_range, front_mask);
    Core.inRange(frame_hsv, rear_lower_range, rear_upper_range, rear_mask);

    // in the masks we still have small amounts of noise, noise that can be filtered out by
    // performing an morphological transform to filter out noise that doesn't meet the rect matrix
    // requirements, this helps remove non rectangular pixel groups
    Point anchor = new Point(-1, -1);
    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5), anchor);

    // During the morphological process we perform an OPEN by a CLOSE operation, these are synonymous
    // with erode and dilate, basically shrinking the white pixel groups down and then expanding them again,
    // this is how we reduce most of the noise, after this process we should have a black background with
    // white rectangular pixal groups where rectangles are detected.
    Imgproc.morphologyEx(front_mask, front_mask, Imgproc.MORPH_OPEN, kernel, anchor, 1);
    Imgproc.morphologyEx(front_mask, front_mask, Imgproc.MORPH_CLOSE, kernel, anchor, 3);
    Imgproc.morphologyEx(rear_mask, rear_mask, Imgproc.MORPH_OPEN, kernel, anchor, 1);
    Imgproc.morphologyEx(rear_mask, rear_mask, Imgproc.MORPH_CLOSE, kernel, anchor, 3);

    // Now that we have the masks, we can use them to filter out data from the HSV frame, so that
    // where the black pixels are removed from the HSV layer and the white pixels are left, this
    // produces a HSV image with only the plate areas exposed
    front_mask_hsv = new Mat();
    rear_mask_hsv = new Mat();
    Core.bitwise_and(frame_hsv, frame_hsv, front_mask_hsv, front_mask);
    Core.bitwise_and(frame_hsv, frame_hsv, front_mask_hsv, front_mask);
    Core.bitwise_and(frame_hsv, frame_hsv, rear_mask_hsv, rear_mask);
    Core.bitwise_and(frame_hsv, frame_hsv, rear_mask_hsv, rear_mask);

    // Apply Canny
    double[] thresholds = Utils.estimateCannyThresholds(frame);
    Imgproc.Canny(rear_mask_hsv, frame_edges, thresholds[0], thresholds[1]);

    // Extract Contours
    List<MatOfPoint> contours = new ArrayList<>();
    Imgproc.findContours(frame_edges, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

    // Instead of removing each non interesting element from the contours array we can just loop
    // over the contours and select the ones we are interested in
    double candidateAreaSize = 0; // used to check of a contours is larger than
    RotatedRect numberplateRect = null;
    MatOfPoint2f currentContour = new MatOfPoint2f();
    MatOfPoint2f currentApproxCurve = new MatOfPoint2f();

    for (int index = 0; index < contours.size(); index++) {
      // Convert the input to a MatOfPoint2F in order to process the contour
      contours.get(index).convertTo(currentContour, CvType.CV_32FC2);

      // Calculate the arch length of the contour
      double archLength = Imgproc.arcLength(currentContour, true);

      // Approximate the curvature of the shape
      Imgproc.approxPolyDP(currentContour, currentApproxCurve, archLength * 0.018, true);

      // Here we filter out any contours that do not have exactly 4 corners
      if(currentApproxCurve.total() != 4) continue;

      // Calculate the contour area and remove contours that are too small in size.
      double area = Imgproc.contourArea(currentApproxCurve);
      if(area < 00) continue;

      // We are interested in this contour, if it is larger than the currently selected candiate
      if(numberplateRect == null || area > candidateAreaSize ) {
        // Extract the rotated rect
        numberplateRect = Imgproc.minAreaRect(currentApproxCurve);
        candidateAreaSize = area;
      }
    }

    Mat normalizedPlate = null;

    // Draw the contours over the frame
    if(numberplateRect != null) {
      normalizedPlate = Utils.rotateAndDeskew(frame, numberplateRect);
      if(normalizedPlate.width() > frame.width() || normalizedPlate.height() > frame.height()) {
        return rear_mask;
      }

      Rect tl = new Rect(0,0, Math.min(normalizedPlate.width(), 480), Math.min(normalizedPlate.height(), 640));
      normalizedPlate.copyTo(frame.submat(tl));
    }

    return front_mask;
  }

  @Override
  public Mat getDebugView() {
    return frame_edges;
  }
}
