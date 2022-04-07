package com.hcifuture.datacollection.service;

import android.accessibilityservice.AccessibilityService;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.res.Configuration;
import android.database.ContentObserver;
import android.hardware.display.DisplayManager;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.Display;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Toast;

import androidx.annotation.Nullable;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.NcnnInstance;
import com.hcifuture.datacollection.contextaction.ContextActionLoader;
import com.hcifuture.datacollection.contextaction.LoaderManager;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.shared.NcnnFunction;
import com.hcifuture.shared.communicate.BuiltInActionEnum;
import com.hcifuture.shared.communicate.BuiltInContextEnum;
import com.hcifuture.shared.communicate.SensorType;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.config.RequestConfig;
import com.hcifuture.shared.communicate.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;
import com.hcifuture.shared.communicate.result.RequestResult;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Timer;
import java.util.TimerTask;

import dalvik.system.DexClassLoader;

public class MainService extends AccessibilityService implements ContextListener, ActionListener {
    private Context mContext;
    private Handler mHandler;

    private CustomBroadcastReceiver mBroadcastReceiver;
    private CustomContentObserver mContentObserver;

    private LoaderManager loaderManager;

    // listening
    private final Uri[] listenedURIs = {
            Settings.System.CONTENT_URI,
            Settings.Global.CONTENT_URI,
    };
    private final String [] listenedActions = {
            Intent.ACTION_AIRPLANE_MODE_CHANGED,
            Intent.ACTION_APPLICATION_RESTRICTIONS_CHANGED,
            Intent.ACTION_BATTERY_LOW,
            Intent.ACTION_BATTERY_OKAY,
            Intent.ACTION_BOOT_COMPLETED,
            Intent.ACTION_CONFIGURATION_CHANGED,
            Intent.ACTION_DOCK_EVENT,
            Intent.ACTION_DREAMING_STARTED,
            Intent.ACTION_DREAMING_STOPPED,
            Intent.ACTION_EXTERNAL_APPLICATIONS_AVAILABLE,
            Intent.ACTION_EXTERNAL_APPLICATIONS_UNAVAILABLE,
            Intent.ACTION_HEADSET_PLUG,
            Intent.ACTION_INPUT_METHOD_CHANGED,
            Intent.ACTION_LOCALE_CHANGED,
            Intent.ACTION_LOCKED_BOOT_COMPLETED,
            Intent.ACTION_MEDIA_BAD_REMOVAL,
            Intent.ACTION_MEDIA_BUTTON,
            Intent.ACTION_MEDIA_CHECKING,
            Intent.ACTION_MEDIA_EJECT,
            Intent.ACTION_MEDIA_MOUNTED,
            Intent.ACTION_MEDIA_NOFS,
            Intent.ACTION_MEDIA_REMOVED,
            Intent.ACTION_MEDIA_SCANNER_FINISHED,
            Intent.ACTION_MEDIA_SCANNER_STARTED,
            Intent.ACTION_MEDIA_SHARED,
            Intent.ACTION_MEDIA_UNMOUNTABLE,
            Intent.ACTION_MEDIA_UNMOUNTED,
            Intent.ACTION_MY_PACKAGE_REPLACED,
            Intent.ACTION_PACKAGES_SUSPENDED,
            Intent.ACTION_PACKAGES_UNSUSPENDED,
            Intent.ACTION_PACKAGE_ADDED,
            Intent.ACTION_PACKAGE_CHANGED,
            Intent.ACTION_PACKAGE_DATA_CLEARED,
            Intent.ACTION_PACKAGE_FIRST_LAUNCH,
            Intent.ACTION_PACKAGE_FULLY_REMOVED,
            Intent.ACTION_PACKAGE_NEEDS_VERIFICATION,
            Intent.ACTION_PACKAGE_REMOVED,
            Intent.ACTION_PACKAGE_REPLACED,
            Intent.ACTION_PACKAGE_RESTARTED,
            Intent.ACTION_PACKAGE_VERIFIED,
            Intent.ACTION_POWER_CONNECTED,
            Intent.ACTION_POWER_DISCONNECTED,
            Intent.ACTION_PROVIDER_CHANGED,
            Intent.ACTION_REBOOT,
            Intent.ACTION_SCREEN_OFF,
            Intent.ACTION_SCREEN_ON,
            Intent.ACTION_SHUTDOWN,
            Intent.ACTION_TIMEZONE_CHANGED,
            Intent.ACTION_TIME_CHANGED,
            Intent.ACTION_UID_REMOVED,
            Intent.ACTION_USER_BACKGROUND,
            Intent.ACTION_USER_FOREGROUND,
            Intent.ACTION_USER_PRESENT,
            Intent.ACTION_USER_UNLOCKED,
            // Bluetooth related
            BluetoothDevice.ACTION_ACL_CONNECTED,
            BluetoothDevice.ACTION_ACL_DISCONNECT_REQUESTED,
            BluetoothDevice.ACTION_ACL_DISCONNECTED,
    };

    private static final HashMap<String, Integer> volume = new HashMap<>();
    static {
        // speaker
        volume.put("volume_music_speaker", 0);
        volume.put("volume_ring_speaker", 0);
        volume.put("volume_alarm_speaker", 0);
        volume.put("volume_voice_speaker", 0);
        volume.put("volume_tts_speaker", 0);
        // headset
        volume.put("volume_music_headset", 0);
        volume.put("volume_voice_headset", 0);
        volume.put("volume_tts_headset", 0);
        // headphone
        volume.put("volume_music_headphone", 0);
        volume.put("volume_voice_headphone", 0);
        volume.put("volume_tts_headphone", 0);
        // Bluetooth A2DP
        volume.put("volume_music_bt_a2dp", 0);
        volume.put("volume_voice_bt_a2dp", 0);
        volume.put("volume_tts_bt_a2dp", 0);
    }
    int brightness;

    public MainService() {
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (loaderManager != null) {
            loaderManager.onAccessibilityEvent(event);
        }
    }

    @Override
    public void onInterrupt() {

    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        mContext = this;
        this.mHandler = new Handler(Looper.getMainLooper());
        this.loaderManager = new LoaderManager(this, this, this);
        loaderManager.start();

        // register broadcast receiver
        mBroadcastReceiver = new CustomBroadcastReceiver();
        IntentFilter filter = new IntentFilter();
        for (String action : listenedActions) {
            filter.addAction(action);
        }
        registerReceiver(mBroadcastReceiver, filter);

        // register content observer
        mContentObserver = new CustomContentObserver(new Handler());
        for (Uri uri : listenedURIs) {
            getContentResolver().registerContentObserver(uri, true, mContentObserver);
        }
    }

    @Override
    public boolean onUnbind(Intent intent) {
        // unregister broadcast receiver
        unregisterReceiver(mBroadcastReceiver);
        // unregister content observer
        getContentResolver().unregisterContentObserver(mContentObserver);

        if (loaderManager != null) {
            loaderManager.stop();
        }

        return super.onUnbind(intent);
    }

    @Override
    public void onActionRecognized(ActionResult action) { }

    @Override
    public void onAction(ActionResult action) {
        mHandler.post(() -> {
            Toast.makeText(mContext, action.getAction(), Toast.LENGTH_SHORT).show();
        });
    }

    @Override
    public void onActionSave(ActionResult action) { }

    @Override
    public void onContext(ContextResult context) {
        mHandler.post(() -> Toast.makeText(mContext, context.getContext(), Toast.LENGTH_SHORT).show());
    }

    class CustomBroadcastReceiver extends BroadcastReceiver {

        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            BroadcastEvent event = new BroadcastEvent(action, "", "BroadcastReceive", intent.getExtras());

            switch (action) {
                case Intent.ACTION_CLOSE_SYSTEM_DIALOGS:
                    event.setTag(intent.getStringExtra("reason"));
                    break;
                case Intent.ACTION_CONFIGURATION_CHANGED:
                    Configuration config = getResources().getConfiguration();
                    event.getExtras().putInt("orientation", config.orientation);
                    break;
                case Intent.ACTION_SCREEN_OFF:
                case Intent.ACTION_SCREEN_ON:
                    // ref: https://stackoverflow.com/a/17348755/11854304
                    DisplayManager dm = (DisplayManager) context.getSystemService(Context.DISPLAY_SERVICE);
                    if (dm != null) {
                        Display[] displays = dm.getDisplays();
                        int [] states = new int[displays.length];
                        for (int i = 0; i < displays.length; i++) {
                            states[i] = displays[i].getState();
                        }
                        event.getExtras().putIntArray("display", states);
                    }
                    break;
            }

            if (loaderManager != null) {
                loaderManager.onBroadcastEvent(event);
            }
        }
    }

    class CustomContentObserver extends ContentObserver {
        public CustomContentObserver(Handler handler) {
            super(handler);
        }

        @Override
        public void onChange(boolean selfChange) {
            onChange(selfChange, null);
        }

        @Override
        public void onChange(boolean selfChange, @Nullable Uri uri) {
            BroadcastEvent event = new BroadcastEvent("", "", "ContentChange");


            if (uri == null) {
                event.setAction("uri_null");
            } else {
                event.setAction(uri.toString());

                String database_key = uri.getLastPathSegment();
                String inter = uri.getPathSegments().get(0);
                if ("system".equals(inter)) {
                    event.setTag(Settings.System.getString(getContentResolver(), database_key));
                } else if ("global".equals(inter)) {
                    event.setTag(Settings.Global.getString(getContentResolver(), database_key));
                }

                int value = Settings.System.getInt(getContentResolver(), database_key, 0);

                // record special information
                if (Settings.System.SCREEN_BRIGHTNESS.equals(database_key)) {
                    // record brightness value difference and update
                    int diff = value - brightness;
                    event.getExtras().putInt("diff", diff);
                    brightness = value;
                    // record brightness mode
                    int mode = Settings.System.getInt(getContentResolver(), Settings.System.SCREEN_BRIGHTNESS_MODE, -1);
                    if (mode == Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL) {
                        event.getExtras().putString("mode", "man");
                    } else if (mode == Settings.System.SCREEN_BRIGHTNESS_MODE_AUTOMATIC) {
                        event.getExtras().putString("mode", "auto");
                    } else {
                        event.getExtras().putString("mode", "unknown");
                    }
                }
                if (database_key.startsWith("volume_")) {
                    if (!volume.containsKey(database_key)) {
                        // record new volume value
                        volume.put(database_key, value);
                    }
                    // record volume value difference and update
                    int diff = value - volume.get(database_key);
                    event.getExtras().putInt("diff", diff);
                    volume.put(database_key, value);
                }
            }

            if (loaderManager != null) {
                loaderManager.onBroadcastEvent(event);
            }
        }
    }

    @Override
    protected boolean onKeyEvent(KeyEvent event) {
        if (loaderManager != null) {
            BroadcastEvent bc_event = new BroadcastEvent("KeyEvent://"+event.getAction()+"/"+event.getKeyCode(), "", "KeyEvent");
            bc_event.getExtras().putInt("action", event.getAction());
            bc_event.getExtras().putInt("code", event.getKeyCode());
            bc_event.getExtras().putInt("source", event.getSource());
            bc_event.getExtras().putLong("eventTime", event.getEventTime());
            bc_event.getExtras().putLong("downTime", event.getDownTime());
            loaderManager.onBroadcastEvent(bc_event);
        }
        return super.onKeyEvent(event);
    }
}