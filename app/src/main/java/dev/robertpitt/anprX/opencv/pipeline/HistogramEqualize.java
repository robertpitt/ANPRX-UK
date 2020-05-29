package dev.robertpitt.anprX.opencv.pipeline;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class HistogramEqualize implements Step<Mat, Mat> {

  @Override
  public Mat execute(Mat value) {
    Imgproc.equalizeHist(value, value);
    return value;
  }
}
