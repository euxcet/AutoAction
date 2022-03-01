package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;

import com.example.datacollection.BuildConfig;
import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.ui.adapter.TaskAdapter;
import com.example.datacollection.utils.NetworkUtils;
import com.example.datacollection.utils.bean.StringListBean;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.FileInputStream;

public class AddTaskActivity extends AppCompatActivity {
    private AppCompatActivity mActivity;
    private Context mContext;

    private TaskList taskList;

    private EditText nameEditText;
    private EditText timesEditText;
    private EditText durationEditText;
    private CheckBox videoCheckbox;
    private CheckBox audioCheckbox;

    private Button confirmButton;
    private Button cancelButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_task);

        mActivity = this;
        mContext = this;

        nameEditText = findViewById(R.id.addTaskNameEdit);
        timesEditText = findViewById(R.id.addTaskTimesEdit);
        durationEditText = findViewById(R.id.addTaskDurationEdit);
        videoCheckbox = findViewById(R.id.addTaskVideoCheckbox);
        audioCheckbox = findViewById(R.id.addTaskAudioCheckbox);

        confirmButton = findViewById(R.id.addTaskConfirmButton);

        cancelButton = findViewById(R.id.addTaskCancelButton);
        cancelButton.setOnClickListener((v) -> this.finish());
        confirmButton.setOnClickListener((v) -> {
            addNewTask();
        });
    }

    private void addNewTask() {
        NetworkUtils.getAllTaskList(mContext, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                StringListBean taskLists = new Gson().fromJson(response.body(), StringListBean.class);
                if (taskLists.getResult().size() > 0) {
                    String taskListId = taskLists.getResult().get(0);
                    NetworkUtils.getTaskList(mContext, taskListId, 0, new StringCallback() {
                        @Override
                        public void onSuccess(Response<String> response) {
                            taskList = new Gson().fromJson(response.body(), TaskList.class);

                            // taskList = TaskList.parseFromLocalFile();
                            TaskList.Task newTask = new TaskList.Task(
                                    TaskList.generateRandomTaskId(),
                                    nameEditText.getText().toString(),
                                    Integer.parseInt(timesEditText.getText().toString()),
                                    Integer.parseInt(durationEditText.getText().toString()),
                                    videoCheckbox.isChecked(),
                                    audioCheckbox.isChecked());
                            taskList.addTask(newTask);
                            // TaskList.saveToLocalFile(taskList);

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
        });

    }
}