package dev.robertpitt.anprX.activities.MainActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;

import com.google.common.util.concurrent.ListenableFuture;
import com.googlecode.tesseract.android.TessBaseAPI;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.os.Looper;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.util.Size;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import dev.robertpitt.anprX.R;
import dev.robertpitt.anprX.activities.SettingsActivity.SettingsActivity;
import dev.robertpitt.anprX.opencv.NumberplateDetectorV2;
import dev.robertpitt.anprX.tesseract.TesseractAPI;

/**
 * Main Camera Activity
 */
public class MainActivity extends AppCompatActivity {
  /**
   * Log Tag, used to tag logs so we can easily file them using logcat.
   */
  private final static String TAG = "ANPRX::MainActivity";

  /**
   * Camera Permission Message ID, this is used for async communication with
   * the permission requester process, which is spawned if we have missing permissions.
   */
  private final static int CAMERA_PERMISSION_REQUEST = 0x01;

  /**
   * Static list of permissions this activity requires, this list will
   * be compared to the actual authorized permissions and if we are missing a
   * invocation to the android permission layer to prompt the user for those permissions.
   */
  private final static String[] permissions = {
      Manifest.permission.CAMERA,
      Manifest.permission.WRITE_EXTERNAL_STORAGE,
      Manifest.permission.VIBRATE
  };

  /**
   * Get the vibrator service
   */
  Vibrator vibrator;

  /**
   * Numberplate Localisation
   */
  private NumberplateDetectorV2 detector = new NumberplateDetectorV2();

  /**
   * OCR API
   */
  private TesseractAPI tesseractAPI = new TesseractAPI(this);

  /**
   * Callback Handler for when OpenCV is loaded and ready to be used.
   */
  private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
    @Override
    public void onManagerConnected(int status) {
      if(status == LoaderCallbackInterface.SUCCESS) {
        Log.i("OpenCV", "OpenCV loaded successfully");
      } else {
        super.onManagerConnected(status);
      }
    }
  };

  /**
   * Camera Instance
   */
  private Camera camera;

  /**
   * Camera Provider Instance
   */
  private ProcessCameraProvider cameraProvider;

  /**
   * Camera Provider Future
   */
  private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

  /**
   * Camera Selector Config
   */
  private CameraSelector cameraSelector = new CameraSelector.Builder()
      .requireLensFacing(CameraSelector.LENS_FACING_BACK)
      .build();

  /**
   * Preview Config
   * This needs to be initialised after the camera is ready
   */
  private Preview preview;

  /**
   * Image Analysis Use Case
   */
  private ImageAnalysis imageAnalysis;

  /**
   * Executor thread, used for background image processing
   */
  private Executor analysisExecutor = Executors.newSingleThreadExecutor();

  /**
   * UI Component for the Toolbar
   */
  private Toolbar toolbar;

  /**
   * Preview View
   */
  private PreviewView previewView;

  /**
   * UI Component for drawing detector results for debugging purposes.
   */
  private ImageView imageOverlayView;
  private ImageButton scanButton;
  private TextView lastVNPTextView;

  /**
   * Handle the result of a permission request, this is the response of the users
   * interaction with the Allow/Deny dialog.
   */
  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == CAMERA_PERMISSION_REQUEST) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        initialiseCamera();
      } else {
        Toast.makeText(this,  "Camera permission was not granted", Toast.LENGTH_LONG).show();
      }
    }
  }

  /**
   * Initialise the options menu in the upper right of the activity
   * @param menu
   * @return
   */
  @Override
  public boolean onCreateOptionsMenu(Menu menu) {
    getMenuInflater().inflate(R.menu.main_menu, menu);
    return super.onCreateOptionsMenu(menu);
  }

  /**
   * Handle when a menu item is selected
   * @param item
   * @return
   */
  @Override
  public boolean onOptionsItemSelected(MenuItem item) {
    switch(item.getItemId()) {
      case R.id.menu_settings_item:
        startActivity(new Intent(this, SettingsActivity.class));
        break;
    }
    return super.onOptionsItemSelected(item);
  }

  /**
   * Activity Creation Handler
   * @param savedInstanceState
   */
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    /**
     * Execute the parent onCreate command to initialise the activity.
     */
    super.onCreate(savedInstanceState);

    /**
     * Bind Services
     */
    vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

    /**
     * Bind View components
     */
    bindViews();

    /**
     * Request Permissions (Once granted notification is received we bind the camera)
     */
    ActivityCompat.requestPermissions(this, permissions, CAMERA_PERMISSION_REQUEST);

    /**
     * Initialise the Tesseract library
     * @see https://tesseract-ocr.github.io/tessdoc/ImproveQuality#dictionaries-word-lists-and-patterns
     */
    tesseractAPI.init("eng", TessBaseAPI.OEM_LSTM_ONLY);
    tesseractAPI.setVariable("tessedit_char_whitelist", " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    tesseractAPI.setVariable("load_system_dawg", "0");
    tesseractAPI.setVariable("load_freq_dawg", "0");
    tesseractAPI.setPageSegMode(TessBaseAPI.PageSegMode.PSM_SINGLE_BLOCK);
  }

  /**
   * This is usually performed during the onCreate phase
   */
  private void bindViews() {
    /**
     * Configure the primary view to display
     */
    setContentView(R.layout.activity_main);

    /**
     * Toolbar setup
     */
    toolbar = findViewById(R.id.toolbar);

    /**
     * Camera Preview Area setup
     */
    previewView = findViewById(R.id.preview_view);

    /**
     * Misc
     */
    imageOverlayView = findViewById(R.id.imageOverlayView);
    scanButton = findViewById(R.id.scan_button);

    lastVNPTextView = findViewById(R.id.lastVNP);
    lastVNPTextView.setVisibility(View.VISIBLE);
    lastVNPTextView.setZ(100);
    lastVNPTextView.setText("------");

    // Configure action support for legacy devices
    setSupportActionBar(toolbar);

    // Bind Event handlers
    scanButton.setOnTouchListener((view, motionEvent) -> this.onScanTouch(view, motionEvent));
  }

  /**
   * Handlde the UI event for the camera action button.
   * @param view
   * @param motionEvent
   * @return boolean
   */
  private boolean onScanTouch(View view, MotionEvent motionEvent) {
    switch (motionEvent.getAction()) {
      case MotionEvent.ACTION_DOWN:
      case MotionEvent.ACTION_UP:
        break;
    }

    return true;
  }

  /**
   * Occurs when the activity is resumed.
   */
  @Override
  protected void onResume() {
    super.onResume();
    if (!OpenCVLoader.initDebug()) {
      OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
    } else {
      System.loadLibrary("opencv_anpr");
      mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }
  }

  /**
   * Bind the camera
   */
  private void initialiseCamera() {
    /**
     * Initialize the provider instance
     */
    cameraProviderFuture = ProcessCameraProvider.getInstance(this);

    /**
     * Listen for provider initialisation success event
     */
    cameraProviderFuture.addListener(() -> {
      try {
        /**
         * get the camera instance with the configured use cases.
         */
        cameraProvider = cameraProviderFuture.get();

        /**
         * Initialize the preview UseCase
         */
        preview = buildPreviewUseCase();

        /**
         * Configure our Image Analysis pipeline for detecting numbers
         */
        imageAnalysis = buildImageAnalysisUseCase();

        /**
         * Bind touch events on the surface to the metering actions
         */
        previewView.setOnTouchListener(this::performMetering);

        /**
         * Attach use cases to the camera with the same lifecycle owner
         */
        camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

        /**
         * Connect the preview use case to the previewView
         */
        preview.setSurfaceProvider(previewView.createSurfaceProvider(camera.getCameraInfo()));

        /**
         * Attach event listener to the view finder to allow touch to focus
         */
        previewView.setOnTouchListener(this::performMetering);
      } catch (ExecutionException | InterruptedException e) {
        // No errors need to be handled for this Future, This should never be reached.
      }
    }, ContextCompat.getMainExecutor(this));
  }

  private boolean performMetering(View view, MotionEvent motionEvent) {
    if (motionEvent.getAction() != MotionEvent.ACTION_UP) {
      MeteringPoint meteringPoint = previewView.createMeteringPointFactory(cameraSelector).createPoint(motionEvent.getX(), motionEvent.getY());
      camera.getCameraControl().startFocusAndMetering(new FocusMeteringAction.Builder(meteringPoint).build());
      return true;
    }
    return false;
  }

  /**
   * Build the preview user case from the camerax library, this will be used to
   * display a video feed of the camera on the screen, this preview will be running on
   * a seperate thread to the numberplate extraction process.
   */
  private Preview buildPreviewUseCase() {
    return new Preview.Builder()
        .setTargetName("anpr-preview")
        .build();
  }

  /**
   * Create the Image Analysis use case to scan for number plates in the background.
   */
  private ImageAnalysis buildImageAnalysisUseCase() {
    /**
     * Create the base configuration
     */
    ImageAnalysis imageAnalysisUseCase = new ImageAnalysis.Builder()
        .setTargetName("anpr-numberplate-detection")
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build();

    /**
     * Connect the analyzer handler to the analysis pipeline.
     */
    imageAnalysisUseCase.setAnalyzer(analysisExecutor, this::analyzeFrame);

    /**
     * Return the use case
     */
    return imageAnalysisUseCase;
  }

  /**
   * Perform Analysis on the input frame (not running on UI thread)
   */
  private void analyzeFrame(ImageProxy image) {
    /**
     * Perform Detection
     */
    try {
      Mat result = detector.detect(image);

      // Show Debug Frame
      Mat debugMat = detector.getDebugView();
      final Bitmap debugBitmap = Bitmap.createBitmap(debugMat.width(), debugMat.height(), Bitmap.Config.ARGB_8888);
      org.opencv.android.Utils.matToBitmap(debugMat, debugBitmap);
      debugMat.release();
      runOnUiThread(() -> {
        imageOverlayView.setImageBitmap(debugBitmap);
        imageOverlayView.setVisibility(View.VISIBLE);
      });

      if(result != null && result.width() > 0 && result.height() > 0) {
        final Bitmap bitmap1 = Bitmap.createBitmap(result.width(), result.height(), Bitmap.Config.ARGB_8888);
        org.opencv.android.Utils.matToBitmap(result, bitmap1);
        result.release();

        tesseractAPI.setImage(bitmap1);
        final String reg = tesseractAPI.getUTF8Text();
        final int confidence = tesseractAPI.meanConfidence();
        runOnUiThread(() -> {
          if(confidence > 70 && reg.length() > 0) {
            lastVNPTextView.setText(String.format("%s - %d", reg, confidence));
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
              vibrator.vibrate(VibrationEffect.createOneShot(250, VibrationEffect.DEFAULT_AMPLITUDE));
            } else {
              //deprecated in API 26
              vibrator.vibrate(250);
            }
          }
        });
      }
    } catch (Exception e) {
    }

    image.close();
  }
}
