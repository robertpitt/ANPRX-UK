package dev.robertpitt.anprX.tesseract;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import com.googlecode.tesseract.android.TessBaseAPI;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import dev.robertpitt.anprX.Utils;

public class TesseractAPI extends TessBaseAPI {
  /**
   * Log Tag
   */
  private static final String TAG = "ANPRX::ANPRXApplication";

  /**
   * External Storage Base Location
   */
  public final String EX_STORAGE_PATH = Environment.getExternalStorageDirectory().getAbsolutePath();

  /**
   * Base path for Tesseract storage
   */
  public final String TESS_BASE_PATH = String.format("%s/tesseract", EX_STORAGE_PATH);

  /**
   * Context for
   */
  private final Context mContext;

  /**
   *
   */
  public TesseractAPI(Context context) {
    super();
    mContext = context;
  }

  public void init(String lang, int ocrEngineMode) {
    // Create the folders if they don't exists
    Utils.mkdir(TESS_BASE_PATH);
    Utils.mkdir(String.format("%s/tessdata", TESS_BASE_PATH));

    // copy the eng lang file from assets folder if not exists.
    File f2 = new File(String.format("%s/tessdata/%s.traindata", TESS_BASE_PATH, lang));
    if(!f2.exists()){
      InputStream in = null;
      try {
        in = mContext.getAssets().open( "tessdata/eng.traineddata");
        FileOutputStream fout = new FileOutputStream(f2);
        byte[] buf = new byte[1024];
        int len;
        while ((len = in.read(buf)) > 0) {
          fout.write(buf, 0, len);
        }
        in.close();
        fout.close();
      } catch (IOException e) {
        Log.e(TAG, e.toString());
        e.printStackTrace();
      }
    }

    super.init(TESS_BASE_PATH, lang, ocrEngineMode);
  }
}
