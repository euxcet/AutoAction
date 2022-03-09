package com.example.datacollection.contextaction;

import android.content.Context;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.widget.Toast;

import com.example.datacollection.contextaction.sensor.AlwaysOnSensorManager;
import com.example.datacollection.contextaction.sensor.ProxSensorManager;
import com.example.ncnnlibrary.communicate.ActionConfig;
import com.example.ncnnlibrary.communicate.ActionListener;
import com.example.ncnnlibrary.communicate.ActionResult;
import com.example.ncnnlibrary.communicate.BuiltInActionEnum;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import dalvik.system.DexClassLoader;

public class ContextActionLoader {
    private Context mContext;

    private ClassLoader classLoader;
    private Class containerClass;

    private Object container;

    private AlwaysOnSensorManager alwaysOnSensorManager;
    private ProxSensorManager proxSensorManager;


    public ContextActionLoader(Context context, DexClassLoader classLoader) {
        this.mContext = context;
        this.classLoader = classLoader;
        try {
            containerClass = classLoader.loadClass("com.example.contextactionlibrary.contextaction.ContextActionContainer");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Object newContainer(List<ActionConfig> config, ActionListener listener) {
        try {
            return containerClass.getDeclaredConstructor(Context.class, List.class, ActionListener.class, boolean.class).newInstance(mContext, config, listener, true);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private void startContainer(Object container) {
        try {
            Method start = containerClass.getMethod("start");
            start.invoke(container);
        } catch (Exception e) {
            e.printStackTrace();

        }
    }

    private void stopContainer(Object container) {
        try {
            Method stop = containerClass.getMethod("stop");
            stop.invoke(container);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Method getOnSensorChanged(Object container) {
        try {
            Method onSensorChanged = containerClass.getMethod("onSensorChangedDex", SensorEvent.class);
            return onSensorChanged;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private void startSensorManager(Object container, Method onSensorChanged) {
        alwaysOnSensorManager = new AlwaysOnSensorManager(mContext,
                SensorManager.SENSOR_DELAY_FASTEST,
                "AlwaysOnSensorManager",
                container,
                onSensorChanged
                );

        proxSensorManager = new ProxSensorManager(mContext,
                SensorManager.SENSOR_DELAY_FASTEST,
                "ProxSensorManager",
                container,
                onSensorChanged
        );
    }

    public void startDetection(List<ActionConfig> config, ActionListener actionListener) {
        try {
            container = newContainer(config, actionListener);
            Method onSensorChanged = getOnSensorChanged(container);
            startSensorManager(container, onSensorChanged);
            startContainer(container);
            alwaysOnSensorManager.start();
            proxSensorManager.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void stopDetection() {
        if (alwaysOnSensorManager != null) {
            alwaysOnSensorManager.stop();
        }
        if (proxSensorManager != null) {
            proxSensorManager.stop();
        }
        if (container != null) {
            stopContainer(container);
        }
    }
}

