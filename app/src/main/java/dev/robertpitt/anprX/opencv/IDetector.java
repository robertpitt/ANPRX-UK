package dev.robertpitt.anprX.opencv;

import androidx.camera.core.ImageProxy;

import org.opencv.core.Mat;

public interface IDetector {
  /**
   *
   */
  Mat detect(Mat src);

  /**
   *
   */
  Mat detect(ImageProxy image);

  Mat getDebugView();
}
