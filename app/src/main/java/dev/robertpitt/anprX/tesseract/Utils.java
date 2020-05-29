package dev.robertpitt.anprX.tesseract;

import java.io.File;

public class Utils {
  /**
   *
   * @param path
   * @return
   */
  public static boolean mkdir(String path) {
    File file = new File(path);
    if(!file.exists()){
      return file.mkdirs();
    }
    return true;
  }
}
