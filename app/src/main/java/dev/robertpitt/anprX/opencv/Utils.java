package dev.robertpitt.anprX.opencv;

import android.graphics.Bitmap;
import android.media.Image;

import androidx.camera.core.ImageProxy;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

public class Utils {

  public static Mat imageToRGB(ImageProxy image) {
    // Extract each planes buffer pointer
    ByteBuffer Y = image.getPlanes()[0].getBuffer();
    ByteBuffer U = image.getPlanes()[1].getBuffer();
    ByteBuffer V = image.getPlanes()[2].getBuffer();

    // Determine size of all 3 planes combined
    int ySize = Y.capacity();
    int uSize = U.capacity();
    int vSize = V.capacity();

    // Create a new byte array that is the same size as the combined planes.
    byte[] nv21 = new byte[ySize + uSize + vSize];

    // Push teh raw byte data into the sink -> (U and V are swapped)
    Y.get(nv21, 0, ySize);
    V.get(nv21, ySize, vSize);
    U.get(nv21, ySize + vSize, uSize);

    // Create a single YUV matrix that contains all planes
    Mat mYuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
    mYuv.put(0, 0, nv21);

    // Now convert the data to RGB
    Mat mRGB = new Mat();
    Imgproc.cvtColor(mYuv, mRGB, Imgproc.COLOR_YUV2RGB_NV21);

    // TODO: Remove this logic from the utils
    Core.rotate(mRGB, mRGB, image.getImageInfo().getRotationDegrees() - 90);

    mYuv.release();
    return mRGB;
  }

  public static Bitmap matToBitmap (Mat src){
    Bitmap bitmap = Bitmap.createBitmap(src.width(), src.height(), Bitmap.Config.ARGB_8888);
    org.opencv.android.Utils.matToBitmap(src, bitmap);
    return bitmap;
  }

  public static double[] estimateCannyThresholds(Mat input) {
    MatOfDouble mu = new MatOfDouble();
    MatOfDouble sigma = new MatOfDouble();
    Core.meanStdDev(input, mu, sigma);
    double lower = Math.max(0, (1.0 - sigma.get(0,0)[0]) * mu.get(0,0)[0]);
    double upper = Math.min(255, (1.0 + sigma.get(0,0)[0]) * mu.get(0,0)[0]);
    sigma.release();
    mu.release();
    return new double[]{lower, upper};
  }

  public static Mat rotateAndDeskew(Mat scene, RotatedRect rect) {
    Mat rotationMat = Imgproc.getRotationMatrix2D(rect.center, rect.angle, 1);
    // Now that we have the rotation matrix, we can apply the geometric transformation using the function warpAffine
    Mat sceneRotated = new Mat();
    Imgproc.warpAffine(scene, sceneRotated, rotationMat, scene.size(), Imgproc.INTER_AREA);
    Mat patch = new Mat();
    Imgproc.getRectSubPix(sceneRotated, rect.size, rect.center, patch);
    sceneRotated.release();

    // If the patch is vertical, drop it to the right
    if(patch.width() < patch.height()) {
      Core.rotate(patch, patch, Core.ROTATE_90_COUNTERCLOCKWISE);
    }

    return patch;
  }

  /**
   * Determine the angle
   * @param pt1
   * @param pt2
   * @param pt0
   * @return
   */
  public static double determineAngle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return ( dx1*dx2 + dy1*dy2 ) / Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
  }

  public static boolean isRectangleInShape(MatOfPoint2f approxCurve) {
    Point[] points = approxCurve.toArray();

    double maxCosine = 0;
    for( int j = 2; j < 5; j++ ) {
      double cosine = Math.abs(determineAngle(points[j % 4], points[j - 2], points[j - 1]));
      maxCosine = Math.max(maxCosine, cosine);
    }

    return maxCosine < 0.3;
  }
}
