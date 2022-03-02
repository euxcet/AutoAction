package com.example.datacollection.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

import com.example.datacollection.utils.bean.TaskListBean;
import com.example.datacollection.utils.FileUtils;
import com.example.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SensorController {
    // sensor
    private SensorManager sensorManager;
    private int samplingPeriod = SensorManager.SENSOR_DELAY_FASTEST;  // fastest
    private Sensor gyroSensor;
    private Sensor linearSensor;
    private Sensor accSensor;
    private Sensor magSensor;
    private Context mContext;
    private List<SensorInfo> sensorData = new ArrayList<>();
    private long lastTimestamp;

    private File saveFile;

    public SensorController(Context context) {
        this.mContext = context;

        sensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);

        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        linearSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        accSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        if (!isSensorSupport()) {
            // Toast.makeText(mContext, "传感器缺失", Toast.LENGTH_LONG).show();
        }
    }

    public void resume() {
        if (sensorManager != null) {
            sensorManager.registerListener(listener, gyroSensor, samplingPeriod);
            sensorManager.registerListener(listener, linearSensor, samplingPeriod);
            sensorManager.registerListener(listener, accSensor, samplingPeriod);
            sensorManager.registerListener(listener, magSensor, samplingPeriod);
        }
    }

    public void pause() {
        if (sensorManager != null) {
            sensorManager.unregisterListener(listener);
        }
    }

    public void start(File file) {
        this.saveFile = file;
        sensorData.clear();
        resume();
    }

    public void stop() {
        pause();
        FileUtils.writeStringToFile(new Gson().toJson(sensorData), this.saveFile);
    }

    public boolean isSensorSupport() {
        return gyroSensor != null && linearSensor != null && accSensor != null && magSensor != null;
    }

    private SensorEventListener listener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            sensorData.add(new SensorInfo(
                    event.sensor.getType(),
                    event.values[0],
                    event.values[1],
                    event.values[2],
                    event.timestamp
            ));
            lastTimestamp = event.timestamp;
        }
        @Override
        public void onAccuracyChanged(Sensor sensor, int i) { }
    };

    public long getLastTimestamp() {
        return lastTimestamp;
    }

    public void upload(String taskListId, String taskId, String subtaskId, String recordId, long timestamp) {
        if (saveFile != null) {
            NetworkUtils.uploadRecordFile(mContext, saveFile, TaskListBean.FILE_TYPE.SENSOR.ordinal(), taskListId, taskId, subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                }
            });
        }
    }
}
