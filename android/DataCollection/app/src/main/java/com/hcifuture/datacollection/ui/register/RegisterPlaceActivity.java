package com.hcifuture.datacollection.ui.register;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.data.CameraController;
import com.hcifuture.datacollection.inference.ImuEventListener;
import com.hcifuture.datacollection.inference.ImuSensorManager;

public class RegisterPlaceActivity extends AppCompatActivity implements ImuEventListener {
    private ImuSensorManager mImuSensorManager;
    private CameraController mCameraController;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register_place);
        mImuSensorManager = new ImuSensorManager(this);
        mImuSensorManager.addListener(this);
        mCameraController = new CameraController(this);
        mCameraController.openCamera(1, false);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mImuSensorManager.stop();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mImuSensorManager.start();
    }

    @Override
    public void onStatus(String status) {
        if (status.equals("OnTable")) {
        }
    }

    @Override
    public void onAction(String action) {
        if (action.equals("PlaceOnTable")) {
            mCameraController.capture();
        }
    }
}