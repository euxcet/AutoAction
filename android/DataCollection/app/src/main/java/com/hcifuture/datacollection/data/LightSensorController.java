package com.hcifuture.datacollection.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import com.google.gson.Gson;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Get data from the ambient light sensor
 * Save data to the file and upload to the backend.
 */
public class LightSensorController {
    private Context mContext;
    private SensorManager mSensorManager;
    private int mSamplingMode = SensorManager.SENSOR_DELAY_FASTEST;
    private Sensor mLightSensor;
    private List<SensorData1D> mSensorData = new ArrayList<>();
    private long mLastTimestamp;

    private File mSensorFile;
    private SensorEventListener mListener;

    /**
     * Init the light sensor service.
     * WARNING: Make sure that the system supports light sensor
     * before calling this constructor.
     * @param context The application context.
     */
    public LightSensorController(Context context) {
        mContext = context;

        mSensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);
        mLightSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);
        mListener = new SensorEventListener() {
            @Override
            public void onSensorChanged(SensorEvent event) {
                mSensorData.add(new SensorData1D(event.values[0], event.timestamp));
                mLastTimestamp = event.timestamp;
            }

            // Not implemented yet
            @Override
            public void onAccuracyChanged(Sensor sensor, int accuracy) {}
        };
    }

    /**
     * Start recording the light sensor data and save to the sensor file.
     * @param file The sensor file to store data.
     */
    public void start(File file) {
        if (!isAvailable()) {
            Log.w("LightSensorController.start()", "Light sensor unavailable!");
            return;
        }
        mSensorFile = file;
        mSensorData.clear();
        if (mSensorManager != null)
            mSensorManager.registerListener(mListener, mLightSensor, mSamplingMode);
    }

    /**
     * Cancel the recording process as if start() has never been called.
     */
    public void cancel() {
        if (!isAvailable()) {
            Log.w("LightSensorController.cancel()", "Light sensor unavailable!");
            return;
        }
        if (mSensorManager != null)
            mSensorManager.unregisterListener(mListener);
        mSensorData.clear();
    }

    /**
     * Called when a whole subtask is recorded.
     * Unregister the listener and write all data to the sensor file.
     */
    public void stop() {
        if (!isAvailable()) {
            Log.w("LightSensorController.stop()", "Light sensor unavailable!");
            return;
        }
        if (mSensorManager != null)
            mSensorManager.unregisterListener(mListener);
        FileUtils.writeLightSensorDataToFile(mSensorData, mSensorFile);
        mSensorData.clear();
    }

    public long getLastTimestamp() { return mLastTimestamp; }

    public boolean isAvailable() {
        return mLightSensor != null;
    }

    public void upload(String taskListId, String taskId, String subtaskId,
                       String recordId, long timestamp) {
        if (!isAvailable()) {
            Log.w("LightSensorController.upload()", "Light sensor unavailable!");
            return;
        }
        if (mSensorFile != null) {
            NetworkUtils.uploadRecordFile(mContext, mSensorFile,
                    TaskListBean.FILE_TYPE.LIGHT.ordinal(), taskListId, taskId,
                    subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) { }
            });
        }
    }
}
