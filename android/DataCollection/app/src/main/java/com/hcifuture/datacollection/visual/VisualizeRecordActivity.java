package com.hcifuture.datacollection.visual;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Color;
import android.hardware.Sensor;
import android.os.Bundle;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.data.SensorInfo;
import com.hcifuture.datacollection.utils.ChartUtils;
import com.hcifuture.datacollection.utils.ColorUtils;
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

import org.checkerframework.checker.units.qual.A;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class VisualizeRecordActivity extends AppCompatActivity {
    private Context mContext;

    private String mTaskListId;
    private String mTaskId;
    private String mSubtaskId;
    private String mRecordId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_visualize_record);
        this.mContext = this;

        Bundle bundle = getIntent().getExtras();
        this.mTaskListId = bundle.getString("taskListId");
        this.mTaskId = bundle.getString("taskId");
        this.mSubtaskId = bundle.getString("subtaskId");
        this.mRecordId = bundle.getString("recordId");
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadRecordViaNetwork();
    }

    private void loadRecordViaNetwork() {
        // TODO: only visualized IMU data currently
        File motionFile = new File(BuildConfig.SAVE_PATH, getTmpFileName(mRecordId, TaskListBean.FILE_TYPE.MOTION.ordinal()));
        File lightFile = new File(BuildConfig.SAVE_PATH, getTmpFileName(mRecordId, TaskListBean.FILE_TYPE.LIGHT.ordinal()));
        if (!motionFile.exists()) {
            NetworkUtils.downloadRecordFile(mContext, mTaskListId, mTaskId, mSubtaskId, mRecordId,
                    TaskListBean.FILE_TYPE.MOTION.ordinal(), new FileCallback() {
                @Override
                public void onSuccess(Response<File> response) {
                    File file = response.body();
                    FileUtils.copy(file, motionFile);
                    visualizeMotionData(motionFile);
                }
            });
        } else visualizeMotionData(motionFile);
        if (!lightFile.exists()) {
            NetworkUtils.downloadRecordFile(mContext, mTaskListId, mTaskId, mSubtaskId, mRecordId,
                    TaskListBean.FILE_TYPE.LIGHT.ordinal(), new FileCallback() {
                        @Override
                        public void onSuccess(Response<File> response) {
                            File file = response.body();
                            FileUtils.copy(file, lightFile);
                            visualizeLightData(lightFile);
                        }
                    });
        } else visualizeLightData(lightFile);
    }

    /**
     * FileType:
     * - 0 timestamp json
     * - 1 motion bin
     * - 2 light bin
     * - 3 audio mp4
     * - 4 video mp4
     */
    private String getTmpFileName(String recordId, int fileType) {
        if (fileType == 0) {
            return "TMP_TIMESTAMP_" + recordId + ".json";
        } else if (fileType == 1) {
            return "TMP_MOTION_" + recordId + ".bin";
        } else if (fileType == 2) {
            return "TMP_LIGHT" + recordId + ".bin";
        } else if (fileType == 3) {
            return "TMP_AUDIO" + recordId + ".mp4";
        } else if (fileType == 4) {
            return "TMP_VIDEO" + recordId + ".mp4";
        }
        return "TMP_UNKNOWN_" + recordId;
    }

    /**
     * Visualize motion data from data file acquired from backend.
     * @param file The motion .bin file from backend.
     */
    private void visualizeMotionData(File file) {
        Map<String, Map<String, List<Double>>> motionData = FileUtils.loadMotionData(file);

        LineChart accChart = findViewById(R.id.visual_acc_chart);
        visualizeSensorData3D(accChart, motionData.get("acc_data"), "Acc");

        LineChart magChart = findViewById(R.id.visual_mag_chart);
        visualizeSensorData3D(magChart, motionData.get("mag_data"), "Mag");

        LineChart gyroChart = findViewById(R.id.visual_gyro_chart);
        visualizeSensorData3D(gyroChart, motionData.get("gyro_data"), "Gyro");

        LineChart linearAccChart = findViewById(R.id.visual_linear_acc_chart);
        visualizeSensorData3D(linearAccChart, motionData.get("linear_acc_data"), "LinearAcc");
    }

    /**
     * Visualize light data from data file acquired from backend.
     * @param file The motion .bin file from backend.
     */
    private void visualizeLightData(File file) {
        Map<String, List<Double>> lightData = FileUtils.loadLightData(file);

        LineChart lightChart = findViewById(R.id.visual_light_chart);
        visualizeSensorData1D(lightChart, lightData, "light");
    }

    /**
     * Visualize a sequence of 1D sensor data on to the line chart.
     * @param chart The line chart variable.
     * @param data  The 1D sensor data, with the following structure:
     *              {"v": [...], "t": [...]}, both lists have the same length.
     * @param label Sensor type label.
     */
    private void visualizeSensorData1D(LineChart chart, Map<String, List<Double>> data, String label) {
        ArrayList<Entry> entries = new ArrayList<>();
        List<Double> v = data.get("v"); List<Double> t = data.get("t");
        DataSetConfig config = new DataSetConfig(label, ColorUtils.GREEN);
        double base = t.get(0);
        int size = t.size();
        for (int i = 0; i < size; i++) {
            entries.add(new Entry((float) ((t.get(i)-base) / 1e9), (float) (double) v.get(i)));
        }

        if (chart.getData() != null && chart.getData().getDataSetCount() > 0) {
            LineDataSet lineDataSet = (LineDataSet) chart.getData().getDataSetByIndex(0);
            lineDataSet.setValues(entries); lineDataSet.notifyDataSetChanged();
            chart.getData().notifyDataChanged();
            chart.notifyDataSetChanged();
        } else {
            ArrayList<ILineDataSet> dataSets = new ArrayList<>();
            LineDataSet dataSet = ChartUtils.createLineDataSet(chart, entries, config);
            dataSets.add(dataSet);
            chart.setData(new LineData(dataSets));
            chart.getData().notifyDataChanged();
            chart.notifyDataSetChanged();
        }
    }

    /**
     * Visualize a sequence of 3D sensor data on to the line chart.
     * @param chart The line chart variable.
     * @param data  The 3D sensor data, with the following structure:
     *              {"x": [...], "y": [...], "z": [...], "t": [...]}, all lists have the same length.
     * @param label Sensor type label.
     */
    private void visualizeSensorData3D(LineChart chart, Map<String, List<Double>> data, String label) {
        ArrayList<Entry> xEntries = new ArrayList<>();
        ArrayList<Entry> yEntries = new ArrayList<>();
        ArrayList<Entry> zEntries = new ArrayList<>();
        List<Double> x = data.get("x"); List<Double> y = data.get("y");
        List<Double> z = data.get("z"); List<Double> t = data.get("t");
        DataSetConfig xConfig = new DataSetConfig(label + "_X", ColorUtils.RED);
        DataSetConfig yConfig = new DataSetConfig(label + "_Y", ColorUtils.GREEN);
        DataSetConfig zConfig = new DataSetConfig(label + "_Z", ColorUtils.BLUE);
        double base = t.get(0);
        int size = t.size();
        for (int i = 0; i < size; i++) {
            xEntries.add(new Entry((float) ((t.get(i)-base) / 1e9), (float) (double) x.get(i)));
            yEntries.add(new Entry((float) ((t.get(i)-base) / 1e9), (float) (double) y.get(i)));
            zEntries.add(new Entry((float) ((t.get(i)-base) / 1e9), (float) (double) z.get(i)));
        }

        if (chart.getData() != null && chart.getData().getDataSetCount() > 0) {
            LineDataSet xLineDataSet = (LineDataSet) chart.getData().getDataSetByIndex(0);
            xLineDataSet.setValues(xEntries); xLineDataSet.notifyDataSetChanged();
            LineDataSet yLineDataSet = (LineDataSet) chart.getData().getDataSetByIndex(1);
            yLineDataSet.setValues(yEntries); yLineDataSet.notifyDataSetChanged();
            LineDataSet zLineDataSet = (LineDataSet) chart.getData().getDataSetByIndex(2);
            zLineDataSet.setValues(zEntries); zLineDataSet.notifyDataSetChanged();
            chart.getData().notifyDataChanged();
            chart.notifyDataSetChanged();
        } else {
            ArrayList<ILineDataSet> dataSets = new ArrayList<>();
            LineDataSet xDataSet = ChartUtils.createLineDataSet(chart, xEntries, xConfig);
            dataSets.add(xDataSet);
            LineDataSet yDataSet = ChartUtils.createLineDataSet(chart, yEntries, yConfig);
            dataSets.add(yDataSet);
            LineDataSet zDataSet = ChartUtils.createLineDataSet(chart, zEntries, zConfig);
            dataSets.add(zDataSet);
            chart.setData(new LineData(dataSets));
            chart.getData().notifyDataChanged();
            chart.notifyDataSetChanged();
        }
    }
}