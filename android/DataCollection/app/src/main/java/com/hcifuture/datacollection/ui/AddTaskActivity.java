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
import com.hcifuture.datacollection.utils.bean.StringListBean;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

/**
 * The activity used for adding a new task by users.
 * Jumped from ConfigTaskActivity.
 */
public class AddTaskActivity extends AppCompatActivity {
    private AppCompatActivity mActivity;
    private Context mContext;

    private TaskListBean mTaskList;

    private EditText mEditTextName;
    private EditText mEditTextTimes;
    private EditText mEditTextDuration;
    private CheckBox mCheckboxVideo;
    private CheckBox mCheckboxAudio;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_task);

        mActivity = this;
        mContext = this;

        mEditTextName = findViewById(R.id.add_task_edit_text_name);
        mEditTextTimes = findViewById(R.id.add_task_edit_text_times);
        mEditTextDuration = findViewById(R.id.add_task_edit_text_duration);
        mCheckboxVideo = findViewById(R.id.add_task_video_switch);
        mCheckboxAudio = findViewById(R.id.audio_switch);

        Button btnAdd = findViewById(R.id.add_task_btn_add);
        Button btnCancel = findViewById(R.id.add_task_btn_cancel);

        btnAdd.setOnClickListener((v) -> addNewTask());
        btnCancel.setOnClickListener((v) -> this.finish());
    }

    private void addNewTask() {
        NetworkUtils.getAllTaskList(mContext, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                StringListBean taskLists = new Gson().fromJson(response.body(), StringListBean.class);
                if (taskLists.getResult().size() > 0) {
                    String taskListId = taskLists.getResult().get(0);
                    NetworkUtils.getTaskList(mContext, GlobalVariable.getInstance().getString("taskListId"), 0, new StringCallback() {
                        @Override
                        public void onSuccess(Response<String> response) {
                            mTaskList = new Gson().fromJson(response.body(), TaskListBean.class);

                            // taskList = TaskList.parseFromLocalFile();
                            TaskListBean.Task newTask = new TaskListBean.Task(
                                    RandomUtils.generateRandomTaskId(),
                                    mEditTextName.getText().toString(),
                                    Integer.parseInt(mEditTextTimes.getText().toString()),
                                    Integer.parseInt(mEditTextDuration.getText().toString()),
                                    mCheckboxAudio.isChecked(),
                                    mCheckboxVideo.isChecked());
                            mTaskList.addTask(newTask);
                            // TaskList.saveToLocalFile(taskList);

                            NetworkUtils.updateTaskList(mContext, mTaskList, 0, new StringCallback() {
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