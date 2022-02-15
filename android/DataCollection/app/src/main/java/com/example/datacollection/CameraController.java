package com.example.datacollection;

import android.Manifest;
import android.content.pm.PackageManager;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.FallbackStrategy;
import androidx.camera.video.FileOutputOptions;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.util.Arrays;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutionException;

public class CameraController {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private Recorder recorder;
    private Recording recording;
    private Preview preview;
    private CameraSelector cameraSelector;
    private VideoCapture videoCapture;
    private ProcessCameraProvider mCameraProvider;
    private AppCompatActivity mActivity;

    public CameraController(AppCompatActivity activity) {
        mActivity = activity;
    }

    public void initialize(boolean open) {
        previewView = mActivity.findViewById(R.id.previewView);
        cameraProviderFuture = ProcessCameraProvider.getInstance(mActivity);
        cameraProviderFuture.addListener(() -> {
            try {
                mCameraProvider = cameraProviderFuture.get();
                preview = new Preview.Builder().build();
                cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                QualitySelector qualitySelector = QualitySelector.fromOrderedList(Arrays.asList(
                        Quality.SD
                ), FallbackStrategy.lowerQualityOrHigherThan(Quality.SD));
                recorder = new Recorder.Builder().setQualitySelector(qualitySelector).build();
                videoCapture = VideoCapture.withOutput(recorder);
                if (open) {
                    Camera camera = mCameraProvider.bindToLifecycle((LifecycleOwner) mActivity,
                            cameraSelector, preview, videoCapture);
                    previewView.setVisibility(View.VISIBLE);
                }
            } catch (ExecutionException | InterruptedException ignored) {
            }
        }, ContextCompat.getMainExecutor(mActivity));
    }

    public void openCamera() {
        if (mCameraProvider == null) {
            initialize(true);
        }
    }

    public void closeCamera() {
        if (mCameraProvider != null) {
            mCameraProvider.unbindAll();
        }
        if (previewView != null) {
            previewView.setVisibility(View.INVISIBLE);
        }
        mCameraProvider = null;
    }

    public void startRecording(File videoFile) {
        if (ActivityCompat.checkSelfPermission(mActivity, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            Log.e("TAG", videoFile.getAbsolutePath());
            recording = recorder.prepareRecording(mActivity, new FileOutputOptions.Builder(videoFile).build())
                    .withAudioEnabled()
                    .start(ContextCompat.getMainExecutor(mActivity), videoRecordEvent -> {
                    });
        }
    }

    public void stopRecording() {
        if (recording != null) {
            recording.stop();
            recording.close();
        }
    }
}
