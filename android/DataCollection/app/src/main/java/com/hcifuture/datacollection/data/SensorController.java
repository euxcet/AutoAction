package com.hcifuture.datacollection.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Get data from four sensors: gyroscope, linear acceleration sensor,
 * accelerometer and magnetic field sensor.
 * Save the sensor data to files and the backend.
 */
public class SensorController {
    // sensor
    private SensorManager sensorManager;
    // SensorManager.SENSOR_DELAY_FASTEST = 0, which is only an enum value
    private int samplingPeriod = SensorManager.SENSOR_DELAY_FASTEST;  // fastest
    private Sensor gyroSensor;
    private Sensor linearSensor;
    private Sensor accSensor;
    private Sensor magSensor;
    private Context mContext;
    private List<SensorInfo> sensorData = new ArrayList<>();
    private long lastTimestamp;

    private File sensorFile;
    private File sensorBinFile;

    /**
     * Constructor.
     * Initialize the four sensors: gyro, linear, acc and mag and check if they are
     * successfully gotten.
     * @param context the current application context.
     */
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

    /**
     * Register the SensorEventListener listener to all sensors.
     */
    public void resume() {
        if (sensorManager != null) {
            sensorManager.registerListener(listener, gyroSensor, samplingPeriod);
            sensorManager.registerListener(listener, linearSensor, samplingPeriod);
            sensorManager.registerListener(listener, accSensor, samplingPeriod);
            sensorManager.registerListener(listener, magSensor, samplingPeriod);
        }
    }

    /**
     * Unregister the listener.
     */
    public void pause() {
        if (sensorManager != null) {
            sensorManager.unregisterListener(listener);
        }
    }

    /**
     * Init the sensorFile and sensorBinFile and call resume() to register the listener
     * @param file the sensorFile
     * @param binFile the sensorBinFile
     */
    public void start(File file, File binFile) {
        sensorFile = file;
        sensorBinFile = binFile;
        sensorData.clear();
        resume();
    }

    /**
     * Unregister the listener and write all data to files.
     */
    public void stop() {
        pause();
        FileUtils.writeStringToFile(new Gson().toJson(sensorData), sensorFile);
        FileUtils.writeIMUDataToFile(sensorData, sensorBinFile);
    }

    /**
     * Check if the four sensors were successfully gotten.
     * @return boolean
     */
    public boolean isSensorSupport() {
        return gyroSensor != null && linearSensor != null && accSensor != null && magSensor != null;
    }

    // this is a private member!
    private SensorEventListener listener = new SensorEventListener() {
        /**
         * Save data in one sampling.
         * Q: Why save all four sensor data to only one sensor data array ???
         * @param event the SensorEvent passed to the listener
         */
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
        /**
         * Not implemented yet.
         * @param sensor
         * @param i
         */
        @Override
        public void onAccuracyChanged(Sensor sensor, int i) { }
    };

    public long getLastTimestamp() {
        return lastTimestamp;
    }

    /**
     * Upload data files to the backend.
     * @param taskListId
     * @param taskId
     * @param subtaskId
     * @param recordId
     * @param timestamp
     */
    public void upload(String taskListId, String taskId, String subtaskId, String recordId, long timestamp) {
        if (sensorFile != null) {
            NetworkUtils.uploadRecordFile(mContext, sensorFile, TaskListBean.FILE_TYPE.SENSOR.ordinal(), taskListId, taskId, subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                }
            });
        }

        if (sensorBinFile != null) {
            NetworkUtils.uploadRecordFile(mContext, sensorBinFile, TaskListBean.FILE_TYPE.SENSOR_BIN.ordinal(), taskListId, taskId, subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                }
            });
        }
    }
}
