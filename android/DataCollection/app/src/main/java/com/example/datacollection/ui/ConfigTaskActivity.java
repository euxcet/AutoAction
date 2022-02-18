package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.ui.adapter.TaskAdapter;

public class ConfigTaskActivity extends AppCompatActivity {
    private ListView taskListView;
    private Button backButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config);

        backButton = findViewById(R.id.taskBackButton);
        backButton.setOnClickListener((v) -> this.finish());

        taskListView = findViewById(R.id.taskListView);
        TaskList taskList = TaskList.parseFromFile(getResources().openRawResource(R.raw.tasklist));

        TaskAdapter taskAdapter = new TaskAdapter(this, taskList);
        taskListView.setAdapter(taskAdapter);
    }
}