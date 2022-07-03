package com.hcifuture.datacollection.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

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
    private SensorManager mSensorManager;
    private int mSamplingMode = SensorManager.SENSOR_DELAY_FASTEST;  // fastest
    private Sensor mGyroSensor;
    private Sensor mLinearSensor;
    private Sensor mAccSensor;
    private Sensor mMagSensor;
    private Context mContext;
    private List<SensorInfo> mSensorData = new ArrayList<>();
    private long mLastTimestamp;

    private File mSensorFile;
    private File mSensorBinFile;
    private SensorEventListener mListener;

    /**
     * Constructor.
     * Initialize the four sensors: gyro, linear, acc and mag and check if they are
     * successfully gotten.
     * @param context the current application context.
     */
    public SensorController(Context context) {
        this.mContext = context;

        mSensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);

        mGyroSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mLinearSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mAccSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mMagSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        mListener = new SensorEventListener() {
            /**
             * Save data in one sampling.
             * Q: Why save all four sensor data to only one sensor data array ???
             * @param event the SensorEvent passed to the listener
             */
            @Override
            public void onSensorChanged(SensorEvent event) {
                mSensorData.add(new SensorInfo(
                        event.sensor.getType(),
                        event.values[0],
                        event.values[1],
                        event.values[2],
                        event.timestamp
                ));
                mLastTimestamp = event.timestamp;
            }
            /**
             * Not implemented yet.
             * @param sensor
             * @param i
             */
            @Override
            public void onAccuracyChanged(Sensor sensor, int i) {}
        };

        if (!isSensorSupport()) {
            Log.w("SensorController", "Sensor missing!");
        }
    }

    /**
     * Called when the user want to start recording IMU data.
     * Init the sensorFile and sensorBinFile and call resume() to register the listener
     * @param file the sensorFile
     * @param binFile the sensorBinFile
     */
    public void start(File file, File binFile) {
        mSensorFile = file;
        mSensorBinFile = binFile;
        mSensorData.clear();
        if (mSensorManager != null) {
            mSensorManager.registerListener(mListener, mGyroSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mLinearSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mAccSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mMagSensor, mSamplingMode);
        }
    }

    /**
     * Called when the user want to cancel an ongoing subtask.
     * Cancel recording the data by unregister the listener.
     */
    public void cancel() {
        if (mSensorManager != null) {
            mSensorManager.unregisterListener(mListener);
        }
        mSensorData.clear();
    }

    /**
     * Called when a whole subtask is recorded.
     * Unregister the listener and write all data to files.
     */
    public void stop() {
        if (mSensorManager != null) {
            mSensorManager.unregisterListener(mListener);
        }
        FileUtils.writeStringToFile(new Gson().toJson(mSensorData), mSensorFile);
        FileUtils.writeIMUDataToFile(mSensorData, mSensorBinFile);
        mSensorData.clear();
    }

    /**
     * Check if the four sensors were successfully gotten.
     * @return boolean
     */
    public boolean isSensorSupport() {
        return mGyroSensor != null && mLinearSensor != null &&
                mAccSensor != null && mMagSensor != null;
    }

    public long getLastTimestamp() {
        return mLastTimestamp;
    }

    /**
     * Upload data files to the backend.
     * @param taskListId
     * @param taskId
     * @param subtaskId
     * @param recordId
     * @param timestamp
     */
    public void upload(String taskListId, String taskId, String subtaskId,
                       String recordId, long timestamp) {
        if (mSensorFile != null) {
            NetworkUtils.uploadRecordFile(mContext, mSensorFile,
                    TaskListBean.FILE_TYPE.SENSOR.ordinal(), taskListId, taskId, subtaskId,
                    recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                }
            });
        }

        if (mSensorBinFile != null) {
            NetworkUtils.uploadRecordFile(mContext, mSensorBinFile,
                    TaskListBean.FILE_TYPE.SENSOR_BIN.ordinal(), taskListId, taskId,
                    subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                }
            });
        }
    }
}
