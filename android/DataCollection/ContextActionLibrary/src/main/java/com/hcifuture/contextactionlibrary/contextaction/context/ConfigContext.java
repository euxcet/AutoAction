package com.hcifuture.contextactionlibrary.contextaction.context;

import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.hardware.display.DisplayManager;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.Display;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.contextactionlibrary.contextaction.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ContextResult;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class ConfigContext extends BaseContext {

    public static String NEED_AUDIO = "context.config.need_audio";
    public static String NEED_NONIMU = "context.config.need_nonimu";
    public static String NEED_SCAN = "context.config.need_scan";

    private String packageName;
    private int brightness;
    private final HashMap<String, Integer> volume;

    private long last_record_all;

    private final LogCollector logCollector;

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
        record_all("stop");
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
        // periodically record_all() every 4 hour
        if (current_call - last_record_all >= 4 * 60 * 60000) {
            record_all("period_4h");
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
                    jsonSilentPut(json, "diff", diff);
                    brightness = value;
                    // record brightness mode
                    int mode = Settings.System.getInt(mContext.getContentResolver(), Settings.System.SCREEN_BRIGHTNESS_MODE, -1);
                    if (mode == Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL) {
                        jsonSilentPut(json, "mode", "man");
                    } else if (mode == Settings.System.SCREEN_BRIGHTNESS_MODE_AUTOMATIC) {
                        jsonSilentPut(json, "mode", "auto");
                    } else {
                        jsonSilentPut(json, "mode", "unknown");
                    }
                    notifyNonIMU(timestamp, "screen brightness change");
                } else if (database_key.startsWith("volume_")) {
                    if (!volume.containsKey(database_key)) {
                        // record new volume value
                        volume.put(database_key, value);
                    }
                    // record volume value difference and update
                    int diff = value - volume.put(database_key, value);
                    jsonSilentPut(json, "diff", diff);
                    notifyAudio(timestamp, "volume change: " + database_key);
                } else if (Settings.Global.BLUETOOTH_ON.equals(database_key) && value == 1) {
                    notifyScan(timestamp, "Bluetooth on");
                } else if (Settings.Global.WIFI_ON.equals(database_key) && value == 1) {
                    notifyScan(timestamp, "Wifi on");
                }
            }
        } else if ("BroadcastReceive".equals(type)) {
            record = true;
            switch (action) {
                case Intent.ACTION_CONFIGURATION_CHANGED:
                    Configuration config = mContext.getResources().getConfiguration();
                    jsonSilentPut(json, "configuration", config.toString());
                    jsonSilentPut(json, "orientation", config.orientation);
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
                        jsonSilentPut(json, "displays", states);
                    }
                    break;
            }
            if (Intent.ACTION_SCREEN_ON.equals(action)) {
                notifyScan(timestamp, "screen on");
            }
        } else if ("KeyEvent".equals(type)) {
            record = true;
            int keycode = extras.getInt("code");
            jsonSilentPut(json, "keycodeString", KeyEvent.keyCodeToString(keycode));

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
                    notifyAudio(timestamp, "key event: " + keycode + " " + KeyEvent.keyCodeToString(keycode));
            }
        }

        if (record) {
            jsonSilentPut(json, "package", packageName);
            for (String key : extras.keySet()) {
                Object obj = JSONObject.wrap(extras.get(key));
                if (obj == null) {
                    obj = JSONObject.wrap(extras.get(key).toString());
                }
                jsonSilentPut(json, key, obj);
            }
            record(timestamp, type, action, tag, json.toString());
        }
    }

    void record(long timestamp, String type, String action, String tag, String other) {
        String line = timestamp + "\t" + type + "\t" + action + "\t" + tag + "\t" + other;
        logCollector.addLog(line);
        Log.e("ConfigContext", "in record");
    }

    void record_all(String action) {
        last_record_all = System.currentTimeMillis();
        JSONObject json = new JSONObject();

        // store brightness
        brightness = Settings.System.getInt(mContext.getContentResolver(), Settings.System.SCREEN_BRIGHTNESS, 0);
        jsonSilentPut(json, "brightness", brightness);

        // store volumes
        for (String key : volume.keySet()) {
            int value = Settings.System.getInt(mContext.getContentResolver(), key, 0);
            volume.put(key, value);
            jsonSilentPut(json, key, value);
        }

        // store configuration and orientation
        Configuration config = mContext.getResources().getConfiguration();
        jsonSilentPut(json, "configuration", config.toString());
        jsonSilentPut(json, "orientation", config.orientation);

        // store system settings
        jsonPutSettings(json, "system", Settings.System.class);

        // store global settings
        jsonPutSettings(json, "global", Settings.Global.class);

        // record
        record(last_record_all, "static", action, "", json.toString());
    }

    static void jsonSilentPut(JSONObject json, String key, Object value) {
        try {
            json.put(key, value);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    void jsonPutSettings(JSONObject json, String key, Class<?> c) {
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
        jsonSilentPut(json, key, jsonArray);
    }

    void notifyAudio(long timestamp, String reason) {
        if (contextListener != null) {
            Log.e("ConfigContext", "broadcast context: " + NEED_AUDIO);
            for (ContextListener listener: contextListener) {
                ContextResult contextResult = new ContextResult(NEED_AUDIO, reason);
                contextResult.setTimestamp(timestamp);
                listener.onContext(contextResult);
            }
        }
    }

    void notifyNonIMU(long timestamp, String reason) {
        if (contextListener != null) {
            Log.e("ConfigContext", "broadcast context: " + NEED_NONIMU);
            for (ContextListener listener: contextListener) {
                ContextResult contextResult = new ContextResult(NEED_NONIMU, reason);
                contextResult.setTimestamp(timestamp);
                listener.onContext(contextResult);
            }
        }
    }

    void notifyScan(long timestamp, String reason) {
        if (contextListener != null) {
            Log.e("ConfigContext", "broadcast context: " + NEED_SCAN);
            for (ContextListener listener: contextListener) {
                ContextResult contextResult = new ContextResult(NEED_SCAN, reason);
                contextResult.setTimestamp(timestamp);
                listener.onContext(contextResult);
            }
        }
    }
}
