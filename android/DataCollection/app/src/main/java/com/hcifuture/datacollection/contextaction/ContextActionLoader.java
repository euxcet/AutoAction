package com.hcifuture.datacollection.contextaction;

import android.content.Context;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.view.KeyEvent;
import android.view.accessibility.AccessibilityEvent;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.contextaction.sensor.IMUSensorManager;
import com.hcifuture.datacollection.contextaction.sensor.ProximitySensorManager;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.config.ContextConfig;
import com.hcifuture.shared.communicate.event.BroadcastEvent;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.ContextListener;
import com.hcifuture.shared.communicate.listener.RequestListener;

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

    private Method onAccessibilityEvent;
    private Method onKeyEvent;

    public ContextActionLoader(Context context, DexClassLoader classLoader) {
        this.mContext = context;
        this.classLoader = classLoader;
        try {
            containerClass = classLoader.loadClass("com.hcifuture.contextactionlibrary.contextaction.ContextActionContainer");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Object newContainer(List<ActionConfig> actionConfig, ActionListener actionListener, List<ContextConfig> contextConfig, ContextListener contextListener, RequestListener requestListener) {
        try {
            return containerClass.getDeclaredConstructor(Context.class,
                    List.class, ActionListener.class,
                    List.class, ContextListener.class,
                    RequestListener.class,
                    boolean.class, boolean.class, String.class)
                    .newInstance(mContext,
                            actionConfig, actionListener,
                            contextConfig, contextListener,
                            requestListener,
                            true, false, BuildConfig.SAVE_PATH);
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
            Method method = containerClass.getMethod("onSensorChangedDex", SensorEvent.class);
            return method;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private Method getOnAccessibilityEvent(Object container) {
        try {
            Method method = containerClass.getMethod("onAccessibilityEventDex", AccessibilityEvent.class);
            return method;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private Method getOnKeyEvent(Object container) {
        try {
            Method method = containerClass.getMethod("onKeyEventDex", BroadcastEvent.class);
            return method;
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

    public void startDetection(List<ActionConfig> actionConfig, ActionListener actionListener, List<ContextConfig> contextConfig, ContextListener contextListener, RequestListener requestListener) {
        try {
            container = newContainer(actionConfig, actionListener, contextConfig, contextListener, requestListener);
            Method onSensorChanged = getOnSensorChanged(container);
            startSensorManager(container, onSensorChanged);
            onAccessibilityEvent = getOnAccessibilityEvent(container);
            onKeyEvent = getOnKeyEvent(container);
            startContainer(container);
            imuSensorManager.start();
            proximitySensorManager.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void startDetection() {
        if (imuSensorManager != null) {
            imuSensorManager.start();
        }
        if (proximitySensorManager != null) {
            proximitySensorManager.start();
        }
        if (container != null) {
            startContainer(container);
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

    public void onAccessibilityEvent(AccessibilityEvent event) {
        try {
            if (onAccessibilityEvent != null) {
                onAccessibilityEvent.invoke(container, event);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void onKeyEvent(KeyEvent event) {
        try {
            if (onKeyEvent != null) {
                onKeyEvent.invoke(container, event);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

