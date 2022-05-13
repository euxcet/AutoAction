package com.hcifuture.contextactionlibrary.contextaction.context;

import android.bluetooth.BluetoothAdapter;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.hardware.display.DisplayManager;
import android.net.Uri;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.Display;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.utils.JSONUtils;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ContextResult;

import org.json.JSONArray;
import org.json.JSONObject;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicInteger;

import androidx.annotation.RequiresApi;

@RequiresApi(api = Build.VERSION_CODES.N)
public class ConfigContext extends BaseContext {

    public static String NEED_AUDIO = "context.config.need_audio";
    public static String NEED_NONIMU = "context.config.need_nonimu";
    public static String NEED_SCAN = "context.config.need_scan";


    private String packageName;
    private int brightness;
    private final HashMap<String, Integer> volume;

    private long last_record_all;

    private final AtomicInteger mLogID = new AtomicInteger(0);

    public ConfigContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener, LogCollector logCollector, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, contextListener, scheduledExecutorService, futureList);
        this.logCollector = logCollector;

        // initialize
        packageName = "";
        brightness = 0;
        volume = new HashMap<>();
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

        last_record_all = 0;
    }

    @Override
    public void start() {
        record_all("start");
    }

    @Override
    public void stop() {
        // do not perform record_all() in stop(),
        // it may cause crashes when frequently called
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {

    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {

    }

    @Override
    public void getContext() {
        long current_call = System.currentTimeMillis();
        // periodically record_all() every 30 min
        if (current_call - last_record_all >= 30 * 60000) {
            record_all("period_30m");
        }
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        CharSequence pkg = event.getPackageName();
        if (pkg != null) {
            packageName = event.getPackageName().toString();
        }
    }

    @Override
    public void onBroadcastEvent(BroadcastEvent event) {
        long timestamp = event.getTimestamp();
        int logID = incLogID();
        String action = event.getAction();
        String type = event.getType();
        String tag = "";
        Bundle extras = event.getExtras();

        boolean record = false;
        JSONObject json = new JSONObject();

        if ("ContentChange".equals(type)) {
            record = true;
            if (!"uri_null".equals(action)) {
                Uri uri = Uri.parse(action);

                String database_key = uri.getLastPathSegment();
                String inter = uri.getPathSegments().get(0);
                if ("system".equals(inter)) {
                    tag = Settings.System.getString(mContext.getContentResolver(), database_key);
                } else if ("global".equals(inter)) {
                    tag = Settings.Global.getString(mContext.getContentResolver(), database_key);
                } else if ("secure".equals(inter)) {
                    tag = Settings.Secure.getString(mContext.getContentResolver(), database_key);
                }

                int value = Settings.System.getInt(mContext.getContentResolver(), database_key, 0);

                // record special information
                if (Settings.System.SCREEN_BRIGHTNESS.equals(database_key)) {
                    // record brightness value difference and update
                    int diff = value - brightness;
                    JSONUtils.jsonPut(json, "diff", diff);
                    brightness = value;
                    // record brightness mode
                    int mode = Settings.System.getInt(mContext.getContentResolver(), Settings.System.SCREEN_BRIGHTNESS_MODE, -1);
                    if (mode == Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL) {
                        JSONUtils.jsonPut(json, "mode", "man");
                    } else if (mode == Settings.System.SCREEN_BRIGHTNESS_MODE_AUTOMATIC) {
                        JSONUtils.jsonPut(json, "mode", "auto");
                    } else {
                        JSONUtils.jsonPut(json, "mode", "unknown");
                    }
                    notifyContext(NEED_NONIMU, timestamp, logID, "screen brightness change");
                } else if (database_key.startsWith("volume_")) {
                    if (!volume.containsKey(database_key)) {
                        // record new volume value
                        volume.put(database_key, value);
                    }
                    // record volume value difference and update
                    int diff = value - volume.put(database_key, value);
                    JSONUtils.jsonPut(json, "diff", diff);
                    notifyContext(NEED_AUDIO, timestamp, logID, "volume change: " + database_key);
                    notifyContext(NEED_SCAN, timestamp, logID, "volume change: " + database_key);
                } else if (Settings.Global.BLUETOOTH_ON.equals(database_key) && value == 1) {
//                    notify(NEED_SCAN, timestamp, logID, "Bluetooth on via global setting");
                } else if (Settings.Global.WIFI_ON.equals(database_key) && value == 2) {
//                    notify(NEED_SCAN, timestamp, logID, "Wifi on via global setting");
                }
            }
        } else if ("BroadcastReceive".equals(type)) {
            record = true;
            switch (action) {
                case Intent.ACTION_CONFIGURATION_CHANGED:
                    Configuration config = mContext.getResources().getConfiguration();
                    JSONUtils.jsonPut(json, "configuration", config.toString());
                    JSONUtils.jsonPut(json, "orientation", config.orientation);
                    break;
                case Intent.ACTION_SCREEN_OFF:
                case Intent.ACTION_SCREEN_ON:
                    // ref: https://stackoverflow.com/a/17348755/11854304
                    DisplayManager dm = (DisplayManager) mContext.getSystemService(Context.DISPLAY_SERVICE);
                    if (dm != null) {
                        Display[] displays = dm.getDisplays();
                        int [] states = new int[displays.length];
                        for (int i = 0; i < displays.length; i++) {
                            states[i] = displays[i].getState();
                        }
                        JSONUtils.jsonPut(json, "displays", states);
                    }
                    break;
            }
            if (Intent.ACTION_SCREEN_ON.equals(action)) {
                notifyContext(NEED_SCAN, timestamp, logID, "screen on");
            } else if (WifiManager.WIFI_STATE_CHANGED_ACTION.equals(action) && extras.getInt(WifiManager.EXTRA_WIFI_STATE) == WifiManager.WIFI_STATE_ENABLED) {
                notifyContext(NEED_SCAN, timestamp, logID, "Wifi on via broadcast");
            } else if (BluetoothAdapter.ACTION_STATE_CHANGED.equals(action) && extras.getInt(BluetoothAdapter.EXTRA_STATE) == BluetoothAdapter.STATE_ON) {
                notifyContext(NEED_SCAN, timestamp, logID, "Bluetooth on via broadcast");
            }
        } else if ("KeyEvent".equals(type)) {
            record = true;
            int keycode = extras.getInt("code");
            JSONUtils.jsonPut(json, "keycodeString", KeyEvent.keyCodeToString(keycode));

            switch (keycode) {
                case KeyEvent.KEYCODE_MEDIA_AUDIO_TRACK:
                case KeyEvent.KEYCODE_MEDIA_CLOSE:
                case KeyEvent.KEYCODE_MEDIA_EJECT:
                case KeyEvent.KEYCODE_MEDIA_FAST_FORWARD:
                case KeyEvent.KEYCODE_MEDIA_NEXT:
                case KeyEvent.KEYCODE_MEDIA_PAUSE:
                case KeyEvent.KEYCODE_MEDIA_PLAY:
                case KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE:
                case KeyEvent.KEYCODE_MEDIA_PREVIOUS:
                case KeyEvent.KEYCODE_MEDIA_RECORD:
                case KeyEvent.KEYCODE_MEDIA_REWIND:
                case KeyEvent.KEYCODE_MEDIA_SKIP_BACKWARD:
                case KeyEvent.KEYCODE_MEDIA_SKIP_FORWARD:
                case KeyEvent.KEYCODE_MEDIA_STEP_BACKWARD:
                case KeyEvent.KEYCODE_MEDIA_STEP_FORWARD:
                case KeyEvent.KEYCODE_MEDIA_STOP:
                case KeyEvent.KEYCODE_MEDIA_TOP_MENU:
                case KeyEvent.KEYCODE_VOLUME_DOWN:
                case KeyEvent.KEYCODE_VOLUME_MUTE:
                case KeyEvent.KEYCODE_VOLUME_UP:
                    notifyContext(NEED_AUDIO, timestamp, logID, "key event: " + KeyEvent.keyCodeToString(keycode));
                    notifyContext(NEED_SCAN, timestamp, logID, "key event: " + KeyEvent.keyCodeToString(keycode));
            }
        }

        if (record) {
            JSONUtils.jsonPut(json, "package", packageName);
            JSONUtils.jsonPutBundle(json, extras);
            record(timestamp, logID, type, action, tag, json.toString());
        }
    }

    private int incLogID() {
        return mLogID.getAndIncrement();
    }

    private void record(long timestamp, int logID, String type, String action, String tag, String other) {
        if (logCollector == null) {
            return;
        }

        String line = timestamp + "\t" + logID + "\t" + type + "\t" + action + "\t" + tag + "\t" + other;
        logCollector.addLog(line);
        Log.e("ConfigContext", "in record");
    }

    private synchronized void record_all(String action) {
        if (logCollector == null) {
            return;
        }

        last_record_all = System.currentTimeMillis();
        int logID = incLogID();
        JSONObject json = new JSONObject();

        // store brightness
        brightness = Settings.System.getInt(mContext.getContentResolver(), Settings.System.SCREEN_BRIGHTNESS, 0);
        JSONUtils.jsonPut(json, "brightness", brightness);

        // store volumes
        for (String key : volume.keySet()) {
            int value = Settings.System.getInt(mContext.getContentResolver(), key, 0);
            volume.put(key, value);
            JSONUtils.jsonPut(json, key, value);
        }

        // store configuration and orientation
        Configuration config = mContext.getResources().getConfiguration();
        JSONUtils.jsonPut(json, "configuration", config.toString());
        JSONUtils.jsonPut(json, "orientation", config.orientation);

        // store system settings
        jsonPutSettings(json, "system", Settings.System.class);

        // store global settings
        jsonPutSettings(json, "global", Settings.Global.class);

        // store secure settings
        jsonPutSettings(json, "secure", Settings.Secure.class);

        // record
        record(last_record_all, logID, "static", action, "", json.toString());
    }

    private void jsonPutSettings(JSONObject json, String key, Class<?> c) {
        JSONArray jsonArray = new JSONArray();
        Field[] fields_glb = c.getFields();
        for (Field f : fields_glb) {
            if (Modifier.isStatic(f.getModifiers())) {
                try {
                    String name = f.getName();
                    Object obj = f.get(null);
                    if (obj != null) {
                        String database_key = obj.toString();
                        Method method = c.getMethod("getString", ContentResolver.class, String.class);
                        String value_s = (String) method.invoke(null, mContext.getContentResolver(), database_key);
                        jsonArray.put(new JSONArray().put(name).put(database_key).put(value_s));
                    }
                } catch (IllegalAccessException | NoSuchMethodException | InvocationTargetException e) {
                    e.printStackTrace();
                }
            }
        }
        JSONUtils.jsonPut(json, key, jsonArray);
    }

    private void notifyContext(String context, long timestamp, int logID, String reason) {
        if (contextListener != null) {
            Log.e("ConfigContext", "broadcast context: " + context);
            for (ContextListener listener: contextListener) {
                ContextResult contextResult = new ContextResult(context, reason);
                contextResult.setTimestamp(timestamp);
                contextResult.getExtras().putInt("logID", logID);
                listener.onContext(contextResult);
            }
        }
    }
}
