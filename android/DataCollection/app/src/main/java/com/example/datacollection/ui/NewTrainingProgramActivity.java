package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;

import com.example.datacollection.R;
import com.example.datacollection.ui.adapter.TrainTaskAdapter;
import com.example.datacollection.utils.GlobalVariable;
import com.example.datacollection.utils.NetworkUtils;
import com.example.datacollection.utils.RandomUtils;
import com.example.datacollection.utils.bean.StringListBean;
import com.example.datacollection.utils.bean.TaskListBean;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class NewTrainingProgramActivity extends AppCompatActivity {
    private Context mContext;
    private AppCompatActivity mActivity;
    private ListView trainTaskListView;
    private TaskListBean taskList;
    private TrainTaskAdapter trainTaskAdapter;
    private String trainId;
    private EditText nameEditText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_new_training_program);
        mContext = this;
        mActivity = this;
        trainTaskListView = findViewById(R.id.trainTaskListView);
        nameEditText = findViewById(R.id.newTrainingNameEdit);
        loadTaskListViaNetwork();

        trainId = RandomUtils.generateRandomTrainId();

        Button confirmButton = findViewById(R.id.trainingProgramConfirmButton);
        confirmButton.setOnClickListener((v) -> {
            if (trainTaskAdapter != null) {
                trainTaskAdapter.getSelected();
                StringBuilder taskIdList = new StringBuilder();
                boolean[] selected = trainTaskAdapter.getSelected();
                for (int i = 0; i < taskList.getTask().size(); i++) {
                    if (selected[i]) {
                        if (taskIdList.toString().equals("")) {
                            taskIdList.append(taskList.getTask().get(i).getId());
                        } else {
                            taskIdList.append(",").append(taskList.getTask().get(i).getId());
                        }
                    }
                }

                NetworkUtils.startTrain(mContext, trainId, nameEditText.getText().toString(), taskList.getId(), taskIdList.toString(), System.currentTimeMillis(),
                        new StringCallback() {
                            @Override
                            public void onSuccess(Response<String> response) {
                                mActivity.finish();
                            }
                        });
            }
        });

        Button cancelButton = findViewById(R.id.trainingProgramCancelButton);
        cancelButton.setOnClickListener((v) -> this.finish());
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadTaskListViaNetwork();
    }

    private void loadTaskListViaNetwork() {
        NetworkUtils.getTaskList(mContext, GlobalVariable.getInstance().getString("taskListId"), 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                taskList = new Gson().fromJson(response.body(), TaskListBean.class);
                trainTaskAdapter = new TrainTaskAdapter(mContext, taskList);
                trainTaskListView.setAdapter(trainTaskAdapter);
            }
        });
    }
}