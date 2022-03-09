package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ListView;

import com.example.datacollection.R;
import com.example.datacollection.ui.adapter.TrainAdapter;
import com.example.datacollection.ui.adapter.TrainTaskAdapter;
import com.example.datacollection.utils.NetworkUtils;
import com.example.datacollection.utils.bean.StringListBean;
import com.example.datacollection.utils.bean.TaskListBean;
import com.example.datacollection.utils.bean.TrainListBean;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class TrainActivity extends AppCompatActivity {
    private static final String TAG = "TrainActivity";
    private Context mContext;
    private ListView trainListView;
    private TrainListBean trainList;
    private TrainAdapter trainAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_train);

        mContext = this;
        Button newProgramButton = findViewById(R.id.newProgramButton);
        newProgramButton.setOnClickListener(view -> {
            Intent intent = new Intent(TrainActivity.this, NewTrainingProgramActivity.class);
            startActivity(intent);
        });

        Button backButton = findViewById(R.id.backButton);
        backButton.setOnClickListener(view -> this.finish());

        trainListView = findViewById(R.id.trainListView);

        loadTrainListViaNetwork();
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadTrainListViaNetwork();
    }

    private void loadTrainListViaNetwork() {
        NetworkUtils.getTrainList(mContext, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                trainList = new Gson().fromJson(response.body(), TrainListBean.class);
                trainAdapter = new TrainAdapter(mContext, trainList);
                trainListView.setAdapter(trainAdapter);
            }
        });
    }
}