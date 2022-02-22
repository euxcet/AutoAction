package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;

public class AddSubtaskActivity extends AppCompatActivity {
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
        setContentView(R.layout.activity_add_subtask);

        nameEditText = findViewById(R.id.addSubtaskNameEdit);
        timesEditText = findViewById(R.id.addSubtaskTimesEdit);
        durationEditText = findViewById(R.id.addSubtaskDurationEdit);
        videoCheckbox = findViewById(R.id.addSubtaskVideoCheckbox);
        audioCheckbox = findViewById(R.id.addSubtaskAudioCheckbox);

        confirmButton = findViewById(R.id.addSubtaskConfirmButton);

        cancelButton = findViewById(R.id.addSubtaskCancelButton);
        cancelButton.setOnClickListener((v) -> this.finish());

        Bundle bundle = getIntent().getExtras();
        int task_id = bundle.getInt("task_id");

        confirmButton.setOnClickListener((v) -> {
            taskList = TaskList.parseFromLocalFile();
            TaskList.Task.Subtask newSubtask = new TaskList.Task.Subtask(
                    0,
                    nameEditText.getText().toString(),
                    Integer.parseInt(timesEditText.getText().toString()),
                    Integer.parseInt(durationEditText.getText().toString()),
                    videoCheckbox.isChecked(),
                    audioCheckbox.isChecked()
            );
            // taskList.getTask()
            // taskList.addTask(newSubtask);
            taskList.getTask().get(task_id).addSubtask(newSubtask);
            taskList.getTask().get(task_id).resetId();
            TaskList.saveToLocalFile(taskList);
            this.finish();
        });
    }
}