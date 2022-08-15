package com.hcifuture.datacollection.ui.register;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Camera;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.action.ActionEnum;
import com.hcifuture.datacollection.action.ActionManager;
import com.hcifuture.datacollection.action.ActionWithObject;
import com.hcifuture.datacollection.action.ObjectDescriptor;
import com.hcifuture.datacollection.data.CameraController;

public class RegisterCaptureActivity extends AppCompatActivity {

    private CameraController mCameraController;
    private ActionManager actionManager;
    private TextView actionListView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register_capture);
        mCameraController = new CameraController(this);
        mCameraController.openCamera(1, false);
        actionManager = ActionManager.getInstance();
        actionListView = findViewById(R.id.actionListView);
        refreshActionList();
        EditText actionNameView = findViewById(R.id.actionNameEditText);
        Button registerButton = findViewById(R.id.registerButton);
        registerButton.setOnClickListener((t) -> {
            mCameraController.capture().whenComplete((feature, e) -> {
                actionManager.register(
                        new ActionWithObject(
                                actionNameView.getText().toString(),
                                getActionEnum(),
                                new ObjectDescriptor(feature)
                        )
                );
                refreshActionList();
            });
        });
    }

    private void refreshActionList() {
        actionListView.setText(ActionManager.encodeActions(
                actionManager.filterWithActionEnum(getActionEnum())));
    }

    private ActionEnum getActionEnum() {
        return ActionEnum.Capture;
    }
}