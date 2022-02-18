package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.ui.adapter.SubtaskAdapter;
import com.example.datacollection.ui.adapter.TaskAdapter;

public class ConfigSubtaskActivity extends AppCompatActivity {
    private ListView subtaskListView;
    private Button backButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config_subtask);

        backButton = findViewById(R.id.subtaskBackButton);
        backButton.setOnClickListener((v) -> this.finish());

        subtaskListView = findViewById(R.id.subtaskListView);
        TaskList taskList = TaskList.parseFromFile(getResources().openRawResource(R.raw.tasklist));

        Bundle bundle = getIntent().getExtras();
        int task_id = bundle.getInt("task_id");

        TextView taskNameView = findViewById(R.id.taskNameView);
        if (taskList != null) {
            taskNameView.setText("Task name: " + taskList.getTask().get(task_id).getName());
        }

        SubtaskAdapter subtaskAdapter = new SubtaskAdapter(this, taskList, task_id);
        subtaskListView.setAdapter(subtaskAdapter);
    }
}