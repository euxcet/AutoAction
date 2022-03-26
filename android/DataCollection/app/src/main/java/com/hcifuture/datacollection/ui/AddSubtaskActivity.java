package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.GlobalVariable;
import com.hcifuture.datacollection.utils.RandomUtils;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class AddSubtaskActivity extends AppCompatActivity {
    private Context mContext;
    private AppCompatActivity mActivity;

    private TaskListBean taskList;

    private EditText nameEditText;
    private EditText timesEditText;
    private EditText durationEditText;
    private CheckBox videoCheckbox;
    private CheckBox audioCheckbox;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_subtask);

        mContext = this;
        mActivity = this;

        nameEditText = findViewById(R.id.addSubtaskNameEdit);
        timesEditText = findViewById(R.id.addSubtaskTimesEdit);
        durationEditText = findViewById(R.id.addSubtaskDurationEdit);
        videoCheckbox = findViewById(R.id.addSubtaskVideoCheckbox);
        audioCheckbox = findViewById(R.id.addSubtaskAudioCheckbox);

        Button confirmButton = findViewById(R.id.addSubtaskConfirmButton);

        Button cancelButton = findViewById(R.id.addSubtaskCancelButton);
        cancelButton.setOnClickListener((v) -> this.finish());

        Bundle bundle = getIntent().getExtras();
        int task_id = bundle.getInt("task_id");

        confirmButton.setOnClickListener((v) -> {
            addNewSubtask(task_id);
        });
    }

    private void addNewSubtask(int task_id) {
        NetworkUtils.getTaskList(mContext, GlobalVariable.getInstance().getString("taskListId"), 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                taskList = new Gson().fromJson(response.body(), TaskListBean.class);

                TaskListBean.Task.Subtask newSubtask = new TaskListBean.Task.Subtask(
                        RandomUtils.generateRandomSubtaskId(),
                        nameEditText.getText().toString(),
                        Integer.parseInt(timesEditText.getText().toString()),
                        Integer.parseInt(durationEditText.getText().toString()),
                        videoCheckbox.isChecked(),
                        audioCheckbox.isChecked()
                );
                taskList.getTask().get(task_id).addSubtask(newSubtask);

                NetworkUtils.updateTaskList(mContext, taskList, 0, new StringCallback() {
                    @Override
                    public void onSuccess(Response<String> response) {
                        mActivity.finish();
                    }
                });
            }
        });
    }
}