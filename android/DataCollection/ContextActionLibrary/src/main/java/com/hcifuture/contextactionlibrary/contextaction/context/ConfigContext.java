package com.hcifuture.contextactionlibrary.contextaction.context;

import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.hardware.SensorEvent;
import android.hardware.display.DisplayManager;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Display;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class ConfigContext extends BaseContext {

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
    String packageName = "";

    public ConfigContext(Context context, ContextConfig config, RequestListener requestListener, List<ContextListener> contextListener) {
        super(context, config, requestListener, contextListener);
    }

    @Override
    public void start() {

    }

    @Override
    public void stop() {

    }

    @Override
    public void getContext() {

    }

    @Override
    public void onIMUSensorChanged(SensorEvent event) {

    }

    @Override
    public void onProximitySensorChanged(SensorEvent event) {

    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        CharSequence pkg = event.getPackageName();
        if (pkg != null) {
            packageName = event.getPackageName().toString();
        }
    }

    static void jsonSilentPut(JSONObject json, String key, Object value) {
        try {
            json.put(key, value);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onBroadcastEvent(BroadcastEvent event) {
        String action = event.getAction();
        String type = event.getType();
        String tag = event.getTag();
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
                    jsonSilentPut(json, "package", packageName);
                }
                if (database_key.startsWith("volume_")) {
                    if (!volume.containsKey(database_key)) {
                        // record new volume value
                        volume.put(database_key, value);
                    }
                    // record volume value difference and update
                    int diff = value - volume.get(database_key);
                    jsonSilentPut(json, "diff", diff);
                    volume.put(database_key, value);
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
        }

        if (record) {
            for (String key : extras.keySet()) {
                Object obj = JSONObject.wrap(extras.get(key));
                if (obj == null) {
                    obj = JSONObject.wrap(extras.get(key).toString());
                }
                jsonSilentPut(json, key, obj);
            }
            record(type, action, tag, json.toString());
        }
    }

    void record(String type, String action, String tag, String other) {
        long cur_timestamp = System.currentTimeMillis();
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(cur_timestamp).append("\t").append(type).append("\t").append(action)
                .append("\t").append(tag).append("\t").append(other);
        String line = stringBuilder.toString();
        // TODO
    }
}
