package com.hcifuture.datacollection.ui.register;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.action.ActionEnum;
import com.hcifuture.datacollection.action.ActionManager;
import com.hcifuture.datacollection.data.CameraCaptureResult;
import com.hcifuture.datacollection.data.CameraController;
import com.hcifuture.datacollection.inference.ImuEventListener;
import com.hcifuture.datacollection.inference.ImuSensorManager;

import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class RegisterPointActivity extends AppCompatActivity implements ImuEventListener {
    private class RegistrationImage {
        public String startImage = null;
        public String endImage = null;
        public String startBackImage = null;
    }
    private enum RegistrationStage {
        IDLE,
        START_POINT,
        END_POINT,
        START_POINT_BACK
    }
    private ImuSensorManager mImuSensorManager;
    private CameraController mCameraController;
    private Vibrator mVibrator;
    private ImageView mStartImageView;
    private ImageView mStartBackImageView;
    private ImageView mEndImageView;
    private TextView mActionListView;
    private Button mRegisterButton;
    private Button mCancelButton;
    private EditText mActionNameView;
    private CameraCaptureResult mCaptureResult;
    private Lock lock = new ReentrantLock();
    private RegistrationStage mRegistrationStage = RegistrationStage.IDLE;

    private RegistrationImage mRegistrationImage = new RegistrationImage();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register_knock);
        mImuSensorManager = new ImuSensorManager(this);
        mImuSensorManager.addListener(this);

        mCameraController = new CameraController(this);
        mCameraController.openCamera(1, false);

        mVibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        mStartImageView = findViewById(R.id.startImageView);
        mStartBackImageView = findViewById(R.id.startBackImageView);
        mEndImageView = findViewById(R.id.endImageView);
        mActionListView = findViewById(R.id.actionListView);
        mRegisterButton = findViewById(R.id.registerButton);
        mCancelButton = findViewById(R.id.cancelButton);
        mCancelButton.setVisibility(View.INVISIBLE);

        mActionNameView = findViewById(R.id.actionNameEditText);

        mCaptureResult = null;

        mRegisterButton.setOnClickListener((v) -> {
            if (mRegisterButton.getText() == "Register") {
                lock.lock();
                try {
                    if (mRegistrationStage == RegistrationStage.IDLE) {
                        mRegistrationStage = RegistrationStage.START_POINT;
                        registerStartingPoint();
                    }
                } finally {
                    lock.unlock();
                }
                mRegisterButton.setText("Done");
                mCancelButton.setVisibility(View.VISIBLE);

            } else {
                // TODO: done
                mRegisterButton.setText("Register");
                mCancelButton.setVisibility(View.INVISIBLE);
            }
            /*
            if (mCaptureResult != null) {
                ActionManager.getInstance().register(new ActionWithObject(
                        mActionNameView.getText().toString(),
                        getActionEnum(),
                        new ObjectDescriptor(mCaptureResult.getFeature())
                ));
                refreshActionList();
            }
             */
        });
        mCancelButton.setOnClickListener((v) -> {
            lock.lock();
            try {
                mRegistrationStage = RegistrationStage.IDLE;
                mRegisterButton.setText("Register");
                mCancelButton.setVisibility(View.INVISIBLE);
            } finally {
                lock.unlock();
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
        if (action.equals("point_out")) {
            lock.lock();
            try {
                if (mRegistrationStage == RegistrationStage.END_POINT) {
                    mVibrator.vibrate(VibrationEffect.createOneShot(200, 128));
                    recordImage(mEndImageView, mRegistrationStage);
                    mRegistrationStage = RegistrationStage.START_POINT_BACK;
                }
            } finally {
                lock.unlock();
            }
        }
        if (action.equals("point_in")) {
            lock.lock();
            try {
                if (mRegistrationStage == RegistrationStage.START_POINT_BACK) {
                    mVibrator.vibrate(VibrationEffect.createOneShot(200, 128));
                    recordImage(mStartBackImageView, mRegistrationStage);
                    mRegistrationStage = RegistrationStage.IDLE;
                }
            } finally {
                lock.unlock();
            }

        }
    }

    private void recordImage(ImageView view, RegistrationStage stage) {
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                mCameraController.capture().whenComplete((result, e) -> {
                    mCaptureResult = result;
                    switch (stage) {
                        case IDLE:
                            break;
                        case START_POINT:
                            mRegistrationImage.startImage = bitmap2Base64(result.getBitmap());
                            break;
                        case END_POINT:
                            mRegistrationImage.endImage= bitmap2Base64(result.getBitmap());
                            break;
                        case START_POINT_BACK:
                            mRegistrationImage.startBackImage = bitmap2Base64(result.getBitmap());
                            break;
                    }

                    runOnUiThread(() -> {
                        view.setImageBitmap(result.getBitmap());
                    });
                });
            }
        }, 300);
    }

    private void registerStartingPoint() {
        mVibrator.vibrate(VibrationEffect.createOneShot(200, 128));
        recordImage(mStartImageView, mRegistrationStage);
        lock.lock();
        try {
            mRegistrationStage = RegistrationStage.END_POINT;
        } finally {
            lock.unlock();
        }
    }

    private void refreshActionList() {
        runOnUiThread(() -> {
            mActionListView.setText(ActionManager.encodeActions(
                    ActionManager.getInstance().getActions()));
        });
    }

    private static String bitmap2Base64(Bitmap bitmap) {
        return byte2Base64(bitmap2Byte(bitmap));
    }

    private static byte[] bitmap2Byte(Bitmap bitmap) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
        return outputStream.toByteArray();
    }

    private static String byte2Base64(byte[] imageByte) {
        if (imageByte == null) {
            return null;
        }
        return Base64.getEncoder().encodeToString(imageByte);
    }

    private ActionEnum getActionEnum() {
        return ActionEnum.Point;
    }
}