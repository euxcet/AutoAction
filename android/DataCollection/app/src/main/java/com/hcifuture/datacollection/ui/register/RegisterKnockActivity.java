package com.hcifuture.datacollection.ui.register;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.action.ActionEnum;
import com.hcifuture.datacollection.action.ActionManager;
import com.hcifuture.datacollection.action.ActionWithObject;
import com.hcifuture.datacollection.action.ObjectDescriptor;
import com.hcifuture.datacollection.data.CameraCaptureResult;
import com.hcifuture.datacollection.data.CameraController;
import com.hcifuture.datacollection.inference.ImuEventListener;
import com.hcifuture.datacollection.inference.ImuSensorManager;

public class RegisterKnockActivity extends AppCompatActivity implements ImuEventListener {
    private ImuSensorManager mImuSensorManager;
    private CameraController mCameraController;
    private Vibrator mVibrator;
    private ImageView mImageView;
    private TextView mActionListView;
    private Button mRegisterButton;
    private EditText mActionNameView;
    private CameraCaptureResult mCaptureResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register_knock);
        mImuSensorManager = new ImuSensorManager(this);
        mImuSensorManager.addListener(this);

        mCameraController = new CameraController(this);
        mCameraController.openCamera(1, false);

        mVibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        mImageView = findViewById(R.id.startImageView);
        mActionListView = findViewById(R.id.actionListView);
        mRegisterButton = findViewById(R.id.registerButton);

        mActionNameView = findViewById(R.id.actionNameEditText);

        mCaptureResult = null;

        mRegisterButton.setOnClickListener((v) -> {
            if (mCaptureResult != null) {
                ActionManager.getInstance().register(new ActionWithObject(
                        mActionNameView.getText().toString(),
                        getActionEnum(),
                        new ObjectDescriptor(mCaptureResult.getFeature())
                ));
                refreshActionList();
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mImuSensorManager != null) {
            mImuSensorManager.stop();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mImuSensorManager != null) {
            mImuSensorManager.start();
        }
        refreshActionList();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mImuSensorManager != null) {
            mImuSensorManager.stop();
        }
    }


    @Override
    public void onStatus(String status) {

    }

    @Override
    public void onAction(String action) {
        if (action.equals("knock")) {
            mVibrator.vibrate(VibrationEffect.createOneShot(200, 128));
            mCameraController.capture().whenComplete((result, e) -> {
                mCaptureResult = result;
                runOnUiThread(() -> {
                    mImageView.setImageBitmap(result.getBitmap());
                });
            });
        }
    }

    private void refreshActionList() {
        runOnUiThread(() -> {
            mActionListView.setText(ActionManager.encodeActions(
                    ActionManager.getInstance().getActions()));
        });
    }

    private ActionEnum getActionEnum() {
        return ActionEnum.Knock;
    }
}