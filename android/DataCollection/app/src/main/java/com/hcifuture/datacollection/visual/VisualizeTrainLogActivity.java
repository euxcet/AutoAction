package com.hcifuture.datacollection.visual;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Color;
import android.os.Bundle;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;
import com.google.gson.Gson;
import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.data.SensorInfo;
import com.hcifuture.datacollection.utils.ChartUtils;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.utils.bean.TrainLogBean;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.model.Response;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class VisualizeTrainLogActivity extends AppCompatActivity {
    private Context mContext;
    private String trainId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_visualize_train_log);
        this.mContext = this;
        Bundle bundle = getIntent().getExtras();
        this.trainId= bundle.getString("trainId");
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadLogViaNetwork();
    }

    private void loadLogViaNetwork() {
        File logFile = new File(BuildConfig.SAVE_PATH, this.trainId + "_log.txt");
        if (!logFile.exists()) {
            NetworkUtils.downloadTrainLog(mContext, trainId, new FileCallback() {
                @Override
                public void onSuccess(Response<File> response) {
                    File file = response.body();
                    FileUtils.copy(file, logFile);
                    visualizeLog(logFile);
                }
            });
        } else {
            visualizeLog(logFile);
        }
    }

    private void visualizeLog(File file) {
        List<SensorInfo> data = new ArrayList<>();
        List<TrainLogBean> logs = new ArrayList<>();
        Gson gson = new Gson();
        try {
            FileInputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis);
            BufferedReader br = new BufferedReader(new InputStreamReader(dis));
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("[D]L")) {
                    TrainLogBean trainLogBean = gson.fromJson(line.substring(4), TrainLogBean.class);
                    logs.add(trainLogBean);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        visualizeValue(findViewById(R.id.lossChart), logs, "Loss", Color.RED);
        visualizeValue(findViewById(R.id.trainAccChart), logs, "Train Accuracy", Color.GREEN);
        visualizeValue(findViewById(R.id.valAccChart), logs, "Validation Accuracy", Color.BLUE);
    }

    private void visualizeValue(LineChart chart, List<TrainLogBean> logs, String label, int color) {
        ArrayList<Entry> value = new ArrayList<>();
        DataSetConfig config = new DataSetConfig(label, color);

        for (int i = 0; i < logs.size(); i++) {
            float v = 0;
            if (label == "Loss") {
                v = logs.get(i).getLoss();
            } else if (label == "Train Accuracy") {
                v = logs.get(i).getTrain_acc();
            } else if (label == "Validation Accuracy") {
                v = logs.get(i).getVal_acc();
            }
            value.add(new Entry(i, v));
        }

        if (chart.getData() != null && chart.getData().getDataSetCount() > 0) {
            LineDataSet lineDataSet = (LineDataSet) chart.getData().getDataSetByIndex(0);
            lineDataSet.setValues(value);
            lineDataSet.notifyDataSetChanged();
            chart.getData().notifyDataChanged();
            chart.notifyDataSetChanged();
        } else {
            ArrayList<ILineDataSet> dataSets = new ArrayList<>();
            LineDataSet dataSet = ChartUtils.createLineDataSet(chart, value, config);
            dataSets.add(dataSet);
            chart.setData(new LineData(dataSets));
        }

    }
}