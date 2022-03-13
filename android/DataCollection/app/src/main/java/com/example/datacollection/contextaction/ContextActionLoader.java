package com.example.datacollection.contextaction;

import android.content.Context;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;

import com.example.datacollection.contextaction.sensor.IMUSensorManager;
import com.example.datacollection.contextaction.sensor.ProximitySensorManager;
import com.example.ncnnlibrary.communicate.config.ActionConfig;
import com.example.ncnnlibrary.communicate.config.ContextConfig;
import com.example.ncnnlibrary.communicate.listener.ActionListener;
import com.example.ncnnlibrary.communicate.listener.ContextListener;

import java.lang.reflect.Method;
import java.util.List;

import dalvik.system.DexClassLoader;

public class ContextActionLoader {
    private Context mContext;

    private ClassLoader classLoader;
    private Class containerClass;

    private Object container;

    private IMUSensorManager imuSensorManager;
    private ProximitySensorManager proximitySensorManager;


    public ContextActionLoader(Context context, DexClassLoader classLoader) {
        this.mContext = context;
        this.classLoader = classLoader;
        try {
            containerClass = classLoader.loadClass("com.example.contextactionlibrary.contextaction.ContextActionContainer");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Object newContainer(List<ActionConfig> actionConfig, ActionListener actionListener, List<ContextConfig> contextConfig, ContextListener contextListener) {
        try {
            return containerClass.getDeclaredConstructor(Context.class,
                    List.class, ActionListener.class,
                    List.class, ContextListener.class,
                    boolean.class, boolean.class)
                    .newInstance(mContext,
                            actionConfig, actionListener,
                            contextConfig, contextListener,
                            true, false);
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
        imuSensorManager = new IMUSensorManager(mContext,
                SensorManager.SENSOR_DELAY_FASTEST,
                "AlwaysOnSensorManager",
                container,
                onSensorChanged
                );

        proximitySensorManager = new ProximitySensorManager(mContext,
                SensorManager.SENSOR_DELAY_FASTEST,
                "ProximitySensorManager",
                container,
                onSensorChanged
        );
    }

    public void startDetection(List<ActionConfig> actionConfig, ActionListener actionListener, List<ContextConfig> contextConfig, ContextListener contextListener) {
        try {
            container = newContainer(actionConfig, actionListener, contextConfig, contextListener);
            Method onSensorChanged = getOnSensorChanged(container);
            startSensorManager(container, onSensorChanged);
            startContainer(container);
            imuSensorManager.start();
            proximitySensorManager.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void stopDetection() {
        if (imuSensorManager != null) {
            imuSensorManager.stop();
        }
        if (proximitySensorManager != null) {
            proximitySensorManager.stop();
        }
        if (container != null) {
            stopContainer(container);
        }
    }
}

