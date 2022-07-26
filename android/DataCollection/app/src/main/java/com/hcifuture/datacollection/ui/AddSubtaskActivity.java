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

/**
 * The activity to add a subtask.
 * Jumped from ConfigSubtaskActivity.
 */
public class AddSubtaskActivity extends AppCompatActivity {
    private Context mContext;
    private AppCompatActivity mActivity;

    private TaskListBean mTaskList;

    private EditText mEditTextName;
    private EditText mEditTextTimes;
    private EditText mEditTextDuration;
    private CheckBox mCheckboxVideo;
    private CheckBox mCheckboxAudio;
    private CheckBox mCheckboxFacing;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_subtask);

        mContext = this;
        mActivity = this;

        mEditTextName = findViewById(R.id.add_subtask_edit_text_name);
        mEditTextTimes = findViewById(R.id.add_subtask_edit_text_times);
        mEditTextDuration = findViewById(R.id.add_subtask_edit_text_duration);
        mCheckboxVideo = findViewById(R.id.add_subtask_video_switch);
        mCheckboxAudio = findViewById(R.id.add_subtask_audio_switch);
        mCheckboxFacing = findViewById(R.id.add_subtask_video_facing);

        Button btnAdd = findViewById(R.id.add_subtask_btn_add);
        Button btnCancel = findViewById(R.id.add_subtask_btn_cancel);

        btnCancel.setOnClickListener((v) -> this.finish());
        Bundle bundle = getIntent().getExtras();
        int task_id = bundle.getInt("task_id");
        btnAdd.setOnClickListener((v) -> {addNewSubtask(task_id);});
    }

    private void addNewSubtask(int task_id) {
        NetworkUtils.getTaskList(mContext, GlobalVariable.getInstance().getString("taskListId"), 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                mTaskList = new Gson().fromJson(response.body(), TaskListBean.class);

                TaskListBean.Task.Subtask newSubtask = new TaskListBean.Task.Subtask(
                        RandomUtils.generateRandomSubtaskId(),
                        mEditTextName.getText().toString(),
                        Integer.parseInt(mEditTextTimes.getText().toString()),
                        Integer.parseInt(mEditTextDuration.getText().toString()),
                        mCheckboxAudio.isChecked(),
                        mCheckboxVideo.isChecked(),
                        mCheckboxFacing.isChecked() ? 0 : 1
                );
                mTaskList.getTasks().get(task_id).addSubtask(newSubtask);

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