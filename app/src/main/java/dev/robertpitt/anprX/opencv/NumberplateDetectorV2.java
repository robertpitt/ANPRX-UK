package dev.robertpitt.anprX.opencv;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static androidx.constraintlayout.widget.Constraints.TAG;

public class NumberplateDetectorV2 implements IDetector {

  /**
   * Initialised Flag
   */
  private boolean _initialised = false;
  private Mat singleChannel8BitImage;
  private Mat processedFrame;
  private Mat edges;
  private Mat normalizedPlate;

  // Filter Class
  private MatOfPoint2f contour2f;
  private MatOfPoint2f approxCurve;
  private MatOfPoint possiblePlateBox;

  private static native void detect_c(long srcAddress);

  /**
   * Initialise Memory Allocations
   */
  protected void _initialise(Mat src) {
    if(!_initialised) {
      singleChannel8BitImage = new Mat();
      edges = new Mat();
      normalizedPlate = null;
      _initialised = true;
    } else {
      singleChannel8BitImage.release();
      processedFrame.release();
      edges.release();
    }
  }
  /**
   * Perform detection on an ImageProxy instance
   */
  public Mat detect(ImageProxy image) {
    return detect(Utils.imageToRGB(image));
  }

  @Override
  public Mat getDebugView() {
    return normalizedPlate;
  }



  @Override
  public Mat detect(Mat rgb) {
    detect_c(rgb.nativeObj);

    _initialise(rgb);

    /**
     * Convert input image to mat, this is the greyscale version of the YUV
     */
    Imgproc.cvtColor(rgb, singleChannel8BitImage, Imgproc.COLOR_RGB2GRAY);

    /**
     * Equalize Histogram
     */
    Imgproc.equalizeHist(singleChannel8BitImage, singleChannel8BitImage);

    /**
     * Do a bilateral filter to clean the noise but keep edges sharp
     */
    processedFrame = new Mat(singleChannel8BitImage.size(), singleChannel8BitImage.type());
    Imgproc.GaussianBlur(singleChannel8BitImage, processedFrame, new Size(5, 5), 3);

    /**
     * Perform a canny edge detection on the image
     */
    double[] thresholds = Utils.estimateCannyThresholds(singleChannel8BitImage);
    Imgproc.Canny(processedFrame, edges, thresholds[0], thresholds[1]);

    List<MatOfPoint> contours = new ArrayList<>();
    Imgproc.findContours(edges, contours, new MatOfPoint(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

    /**
     * Iterate over the contours, skipping contours that we are not interested in.
     */
    List<MatOfPoint> filteredContours = filterContours(contours);

    double candidateAreaSize = 0; // used to check of a contours is larger than
    RotatedRect numberplateRect = null;
    MatOfPoint2f currentContour = new MatOfPoint2f();
    MatOfPoint2f currentApproxCurve = new MatOfPoint2f();

    for (int index = 0; index < filteredContours.size(); index++) {
      /**
       * Convert the input to a MatOfPoint2F in order to process the contour
       */
      filteredContours.get(index).convertTo(currentContour, CvType.CV_32FC2);

      /**
       * alculate the arch length of the contour
       */
      double archLength = Imgproc.arcLength(currentContour, true);

      /**
       * Approximate the curvature of the shape
       */
      Imgproc.approxPolyDP(currentContour, currentApproxCurve, archLength * 0.018, true);

      /**
       * Here we filter out any contours that do not have exactly 4 corners
       */
      if(currentApproxCurve.total() != 4) continue;

      /**
       * Calculate the contour area and remove contours that are too small in size.
       */
      double area = Imgproc.contourArea(currentApproxCurve);
      if(area < 1000) continue;

      /**
       * We are interested in this contour, if it is larger than the currently selected candiate
       */
      if(numberplateRect == null || area > candidateAreaSize) {
        numberplateRect = Imgproc.minAreaRect(currentApproxCurve);
        candidateAreaSize = area;
      }
    }

    // Draw the contours over the frame
    if(numberplateRect != null) {
      normalizedPlate = Utils.rotateAndDeskew(singleChannel8BitImage, numberplateRect);
      if(normalizedPlate.width() > rgb.width() || normalizedPlate.height() > rgb.height()) {
        return null;
      }

      double ratio = normalizedPlate.width() / normalizedPlate.height();
      if(ratio < 2.5) return null;

      Imgproc.rectangle(singleChannel8BitImage, numberplateRect.boundingRect(), new Scalar(255, 255, 255), -1);
//      Imgproc.drawContours(singleChannel8BitImage, filteredContours, index, new Scalar(255, 255, 255), -1);

      Imgproc.threshold(normalizedPlate, normalizedPlate, 100, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
//
//      final Bitmap bitmap1 = Bitmap.createBitmap(normalizedPlate.width(), normalizedPlate.height(), Bitmap.Config.ARGB_8888);
//      org.opencv.android.Utils.matToBitmap(normalizedPlate, bitmap1);
//      tessBaseAPI.setImage(bitmap1);
    }

    return normalizedPlate;
  }

  /**
   * https://books.google.co.uk/books?id=FtCBDwAAQBAJ&pg=PA86&lpg=PA86&dq=opencv+contour+number+plate+filters&source=bl&ots=PxUc2S1fBq&sig=ACfU3U0-l6X1MWagWDwGA5pJghqVkX4GFQ&hl=en&sa=X&ved=2ahUKEwjEjc3Y183pAhVtSBUIHQC-CXAQ6AEwBXoECAoQAQ#v=onepage&q=opencv%20contour%20number%20plate%20filters&f=false
   * untested snipped that may help.
   */
//  public void verifySize(RotatedRect candidate) {
//    double error = 0.4;
//    double aspect = 4.7272;
//    double min = 15 * aspect * 15; // = 15px, this will need to be calibrated
//    double max = 125 * aspect * 125; // 125px = width of plate
//    double rmin = aspect - aspect * error;
//    double rmax = aspect + aspect * error;
//    double area = candidate.size.area();
//    double r = (float)candidate.size.height / (float)candidate.size.width;
//    if(r < 1) {
//      r = 1/r;
//    }
//
//    return !((area < min || area > max) || (r < rmin || r > rmax))
//  }

  private List<MatOfPoint> filterContours(List<MatOfPoint> contours) {
    List<MatOfPoint> results = new ArrayList<>();

    /**
     * Extract the points of the contour (this approach is much faster than {new MatOfPoint2f(mop.toArray())}
     */
    contour2f = new MatOfPoint2f();
    approxCurve = new MatOfPoint2f();

    /**
     * Itterate over the contours, skipping contours that we are not interested in.
     */
    for(int i = 0; i < contours.size(); i++) {
      contours.get(i).convertTo(contour2f, CvType.CV_32F);

      /**
       * Approximate the polygon from the contour
       */
      Imgproc.approxPolyDP(contour2f, approxCurve, Imgproc.arcLength(contour2f, true) * 0.018, true);
      contour2f.release();

      /**
       * Remove those where the total sides of the approximated curve is not rectangle
       */
      if(approxCurve.total() != 4){
        approxCurve.release();
        continue;
      }

      /**
       * Calculate the total area size for the shape so we can filter
       * the selections that are too small or to0 big.
       */
      double areaSize = Math.abs(Imgproc.contourArea(approxCurve));
      if(areaSize < 1000) {
        approxCurve.release();
        continue;
      }

      // Hate this conversion!
      possiblePlateBox = new MatOfPoint(approxCurve.toArray());

      /**
       * Exclude the contour of the approximation is not convex
       *
       * @see https://en.wikipedia.org/wiki/Convex_polygon#Properties
       */
      if(!Imgproc.isContourConvex(possiblePlateBox)) {
        possiblePlateBox.release();
        approxCurve.release();
        continue;
      }

      /**
       * Determine if the shape is rectangular
       */
      if(!Utils.isRectangleInShape(approxCurve)) {
        possiblePlateBox.release();
        approxCurve.release();
        continue;
      }

      approxCurve.release();
      possiblePlateBox.release();
      results.add(new MatOfPoint(contours.get(i)));
    }

    return results;
  }
}
