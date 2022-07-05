package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.GlobalVariable;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.ui.adapter.SubtaskAdapter;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

/**
 * The activity used to configure subtasks.
 * Jumped from the task in TaskAdapter.
 */
public class ConfigSubtaskActivity extends AppCompatActivity {
    private Context mContext;
    private ListView mSubtaskListView;
    private TaskListBean mTaskList;
    private SubtaskAdapter mSubtaskAdapter;
    private TextView mTaskNameView;
    private int mTaskId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config_subtask);
        mContext = this;

        Button backButton = findViewById(R.id.config_subtask_btn_back);
        backButton.setOnClickListener((v) -> this.finish());

        mSubtaskListView = findViewById(R.id.config_subtask_list_view);

        Bundle bundle = getIntent().getExtras();
        mTaskId = bundle.getInt("task_id");

        Button addButton = findViewById(R.id.config_subtask_btn_add_subtask);
        addButton.setOnClickListener((v) -> {
            Bundle addBundle = new Bundle();
            addBundle.putInt("task_id", mTaskId);
            Intent intent = new Intent(ConfigSubtaskActivity.this, AddSubtaskActivity.class);
            intent.putExtras(addBundle);
            startActivity(intent);
        });

        Button configButton = findViewById(R.id.config_subtask_btn_config_task);
        configButton.setOnClickListener((v) -> {
            Bundle configBundle = new Bundle();
            configBundle.putInt("task_id", mTaskId);
            Intent intent = new Intent(ConfigSubtaskActivity.this, ModifyTaskActivity.class);
            intent.putExtras(configBundle);
            startActivity(intent);
        });

        mTaskNameView = findViewById(R.id.config_subtask_title);
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
                mTaskList = new Gson().fromJson(response.body(), TaskListBean.class);
                mTaskNameView.setText("Task: " + mTaskList.getTasks().get(mTaskId).getName());
                mSubtaskAdapter = new SubtaskAdapter(mContext, mTaskList, mTaskId);
                mSubtaskListView.setAdapter(mSubtaskAdapter);
            }
        });
    }
}