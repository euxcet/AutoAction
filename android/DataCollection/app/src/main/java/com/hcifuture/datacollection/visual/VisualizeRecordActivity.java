package com.hcifuture.datacollection.visual;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Color;
import android.hardware.Sensor;
import android.os.Bundle;
import android.util.Log;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.data.SensorInfo;
import com.hcifuture.datacollection.utils.ChartUtils;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class VisualizeRecordActivity extends AppCompatActivity {
    private Context mContext;

    private String taskListId;
    private String taskId;
    private String subtaskId;
    private String recordId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_visualize_record);

        this.mContext = this;

        Bundle bundle = getIntent().getExtras();
        this.taskListId = bundle.getString("taskListId");
        this.taskId = bundle.getString("taskId");
        this.subtaskId = bundle.getString("subtaskId");
        this.recordId = bundle.getString("recordId");

    }

    @Override
    protected void onResume() {
        super.onResume();
        loadRecordViaNetwork();
    }

    private void loadRecordViaNetwork() {
        /*
        - fileType
            - 0 sensor json
            - 1 timestamp json
            - 2 audio mp4
            - 3 video mp4
            - 4 sensor bin
        */
        // IMU
        File imuFile = new File(BuildConfig.SAVE_PATH, getTempFileName(recordId, TaskListBean.FILE_TYPE.SENSOR_BIN.ordinal()));
        if (!imuFile.exists()) {
            NetworkUtils.downloadRecordFile(mContext, taskListId, taskId, subtaskId, recordId, TaskListBean.FILE_TYPE.SENSOR_BIN.ordinal(), new FileCallback() {
                @Override
                public void onSuccess(Response<File> response) {
                    File file = response.body();
                    FileUtils.copy(file, imuFile);
                    visualizeIMU(imuFile);
                }
            });
        } else {
            visualizeIMU(imuFile);
        }
    }

    private String getTempFileName(String recordId, int fileType) {
        if (fileType == 0) {
            return "TEMP_SENSOR_" + recordId + ".json";
        } else if (fileType == 1) {
            return "TEMP_TIMESTAMP_" + recordId + ".json";
        } else if (fileType == 2) {
            return "TEMP_AUDIO_" + recordId + ".mp4";
        } else if (fileType == 3) {
            return "TEMP_VIDEO_" + recordId + ".mp4";
        } else if (fileType == 4) {
            return "TEMP_SENSOR_BIN_" + recordId + ".bin";
        }
        return "TEMP_UNKNOWN_" + recordId;
    }

    private void visualizeIMU(File file) {
        List<SensorInfo> data = FileUtils.loadIMUBinData(file);

        LineChart linearChart = findViewById(R.id.linearChart);
        visualizeSensor(linearChart, data, Sensor.TYPE_LINEAR_ACCELERATION, "LINEAR");

        LineChart gyroChart = findViewById(R.id.gyroChart);
        visualizeSensor(gyroChart, data, Sensor.TYPE_GYROSCOPE, "GYROSCOPE");

        LineChart accChart = findViewById(R.id.accChart);
        visualizeSensor(accChart, data, Sensor.TYPE_ACCELEROMETER, "ACCELERATOR");
    }

    private void visualizeSensor(LineChart chart, List<SensorInfo> data, int sensorType, String labelPrefix) {
        ArrayList<ArrayList<Entry>> valuesList = new ArrayList<>();
        List<DataSetConfig> configList = Arrays.asList(
                new DataSetConfig(labelPrefix + "_X", Color.RED),
                new DataSetConfig(labelPrefix + "_Y", Color.GREEN),
                new DataSetConfig(labelPrefix + "_Z", Color.BLUE)
        );

        for (int i = 0; i < configList.size(); i++) {
            valuesList.add(new ArrayList<>());
        }

        long minTimestamp = Long.MAX_VALUE;
        for (SensorInfo info: data) {
            if (info.getTime() < minTimestamp) {
                minTimestamp = info.getTime();
            }
        }

        for (SensorInfo info: data) {
            List<Float> d = info.getData();
            int idx = d.get(0).intValue();
            float timestampDelta = (info.getTime() - minTimestamp) / 1000000000.0f;
            if (idx == sensorType) {
                valuesList.get(0).add(new Entry(timestampDelta, d.get(1))); // LINEAR_X
                valuesList.get(1).add(new Entry(timestampDelta, d.get(2))); // LINEAR_Y
                valuesList.get(2).add(new Entry(timestampDelta, d.get(3))); // LINEAR_Z
            }
        }

        if (chart.getData() != null && chart.getData().getDataSetCount() > 0) {
            for (int i = 0; i < valuesList.size(); i++) {
                LineDataSet lineDataSet = (LineDataSet) chart.getData().getDataSetByIndex(i);
                lineDataSet.setValues(valuesList.get(i));
                lineDataSet.notifyDataSetChanged();
            }
            chart.getData().notifyDataChanged();
            chart.notifyDataSetChanged();
        } else {
            ArrayList<ILineDataSet> dataSets = new ArrayList<>();
            for(int i = 0; i < valuesList.size(); i++) {
                LineDataSet dataSet = ChartUtils.createLineDataSet(chart, valuesList.get(i), configList.get(i));
                dataSets.add(dataSet);
            }
            chart.setData(new LineData(dataSets));
        }
    }

}