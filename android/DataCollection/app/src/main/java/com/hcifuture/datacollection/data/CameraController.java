package com.hcifuture.datacollection.data;

import static com.lzy.okgo.utils.HttpUtils.runOnUiThread;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.util.Size;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.FallbackStrategy;
import androidx.camera.video.FileOutputOptions;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.common.util.concurrent.ListenableFuture;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The controller for managing the camera data.
 */
public class CameraController {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView cameraPreview;
    private Recorder recorder;
    private Recording recording;
    private Preview preview;
    private CameraSelector cameraSelector;
    private VideoCapture videoCapture;
    private ProcessCameraProvider mCameraProvider;
    private AppCompatActivity mActivity;
    private ExecutorService executorService;
    private File saveFile;
    private AtomicBoolean needCapture;
    private CompletableFuture<float[]> captureFuture;

    public CameraController(AppCompatActivity activity) {
        mActivity = activity;
        executorService = Executors.newFixedThreadPool(2);
        needCapture = new AtomicBoolean(false);
    }

    private Bitmap imageToBitmap(Image image) {
        byte[] data = imageToByteArray(image);
        Bitmap bitmap = BitmapFactory.decodeByteArray(data, 0, data.length);
        return bitmap;
    }

    private byte[] imageToByteArray(Image image) {
        byte[] data = null;
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        data = new byte[buffer.capacity()];
        buffer.get(data);
        return data;
    }

    private Bitmap yuvToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    public Bitmap resizeBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false);
        bm.recycle();
        return resizedBitmap;
    }

    public void initialize(boolean open, int lensFacing, boolean enableCapture) {
        Log.d("CameraController.initialize()", "Camera init called.");
        cameraPreview = mActivity.findViewById(R.id.camera_preview);
        cameraProviderFuture = ProcessCameraProvider.getInstance(mActivity);
        cameraProviderFuture.addListener(() -> {
            try {
                mCameraProvider = cameraProviderFuture.get();
                preview = new Preview.Builder().build();
                cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(lensFacing)
                        .build();
                preview.setSurfaceProvider(cameraPreview.getSurfaceProvider());
                QualitySelector qualitySelector = QualitySelector.fromOrderedList(Arrays.asList(
                        Quality.SD
                ), FallbackStrategy.lowerQualityOrHigherThan(Quality.SD));
                recorder = new Recorder.Builder().setQualitySelector(qualitySelector).build();
                videoCapture = VideoCapture.withOutput(recorder);

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(1280, 720))
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                imageAnalysis.setAnalyzer(executorService, imageProxy -> {
                    @SuppressLint("UnsafeOptInUsageError")
                    Bitmap bitmap = resizeBitmap(yuvToBitmap(imageProxy.getImage()), 224, 224);
                    int bytes = bitmap.getByteCount();
                    ByteBuffer buffer = ByteBuffer.allocate(bytes);
                    bitmap.copyPixelsToBuffer(buffer);
                    byte[] data = buffer.array();
                    // TODO: send data
                    if (needCapture.get() && captureFuture != null) {

                    }
                });

                if (open) {
                    if (enableCapture) {
                        mCameraProvider.bindToLifecycle(mActivity, cameraSelector, videoCapture, preview);
                    } else {
                        mCameraProvider.bindToLifecycle(mActivity, cameraSelector, imageAnalysis, preview);
                    }
                    cameraPreview.setVisibility(View.VISIBLE);
                }
            } catch (ExecutionException | InterruptedException ignored) {
                Log.d("CameraController.initialize()", "Camera init failed!");
            }
        }, ContextCompat.getMainExecutor(mActivity));
    }

    public void openCamera(int lensFacing, boolean enableCapture) {
        if (mCameraProvider == null) {
            initialize(true, lensFacing, enableCapture);
        }
    }

    public void closeCamera() {
        if (mCameraProvider != null) {
            mCameraProvider.unbindAll();
        }
        if (cameraPreview != null) {
            cameraPreview.setVisibility(View.INVISIBLE);
        }
        mCameraProvider = null;
    }

    public void start(File videoFile) {
        if (ActivityCompat.checkSelfPermission(mActivity, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            this.saveFile = videoFile;
            // modified: no audio
            recording = recorder.prepareRecording(mActivity, new FileOutputOptions.Builder(saveFile).build())
                    .start(ContextCompat.getMainExecutor(mActivity), videoRecordEvent -> {});
        }
    }

    public void cancel() {
        stop();
    }

    public void stop() {
        if (recording != null) {
            recording.stop();
            recording.close();
        }
    }

    public CompletableFuture<float[]> capture() {
        if (needCapture.get()) {
            return null;
        }
        captureFuture = new CompletableFuture<>();
        needCapture.set(true);
        return captureFuture;
    }

    public void upload(String taskListId, String taskId, String subtaskId, String recordId, long timestamp) {
        if (saveFile != null) {
            NetworkUtils.uploadRecordFile(mActivity, saveFile, TaskListBean.FILE_TYPE.VIDEO.ordinal(), taskListId, taskId, subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                    Log.d("CameraController.upload() onSuccess()", response.body());
                }

                @Override
                public void onError(Response<String> response) {
                    Log.d("CameraController.upload() onError()", response.toString());
                }
            });
        }
    }
}
