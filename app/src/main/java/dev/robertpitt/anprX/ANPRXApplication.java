package dev.robertpitt.anprX;

import android.app.Application;

import io.sentry.android.core.SentryAndroid;
import io.sentry.core.SentryLevel;

public class ANPRXApplication extends Application {
  /**
   * Log Tag
   */
  private static final String TAG = "ANPRX::ANPRXApplication";

  /**
   * Initialise the Application
   */
  @Override
  public void onCreate() {
    initialiseSentry();
    super.onCreate();
  }

  /**
   * Configure Sentry
   */
  private void initialiseSentry() {
    /**
     * Manual Initialization of the Sentry Android SDK
     * @Context - Instance of the Android Context
     * @Options - Call back function that you need to provide to be able to modify the options.
     * The call back function is provided with the options loaded from the manifest.
     */
    SentryAndroid.init(this, options -> {
      // Add a callback that will be used before the event is sent to Sentry.
      // With this callback, you can modify the event or, when returning null, also discard the event.
      options.setBeforeSend((event, hint) -> {
        if (SentryLevel.DEBUG.equals(event.getLevel()))
          return null;
        else
          return event;
      });
    });
  }
}
