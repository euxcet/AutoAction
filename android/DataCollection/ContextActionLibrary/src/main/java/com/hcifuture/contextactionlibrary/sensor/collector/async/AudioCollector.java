package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;
import android.media.MediaRecorder;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.utils.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class AudioCollector extends AsynchronousCollector {
    private MediaRecorder mMediaRecorder;
    private final AtomicBoolean isCollecting;

    /*
      Error code:
        0: No error
        1: Invalid audio length
        2: Null audio filename
        3: Concurrent task of audio recording
        4: Unknown audio recording exception
        5: Unknown exception when stopping recording
     */

    public AudioCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        isCollecting = new AtomicBoolean(false);
    }

    @Override
    public void initialize() {

    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    public CompletableFuture<Void> collect(TriggerConfig config) {
        CompletableFuture<Void> ft = new CompletableFuture<>();
        if (config.getAudioLength() == 0) {
            ft.complete(null);
            return ft;
        }
        if (saveFile == null) {
            ft.complete(null);
            return ft;
        }
        if (!Objects.requireNonNull(saveFile.getParentFile()).exists()) {
            saveFile.getParentFile().mkdirs();
        }
        if (!isRecording.get()) {
            isRecording.set(true);
            try {
                mMediaRecorder = new MediaRecorder();
                mMediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
                mMediaRecorder.setAudioChannels(2);
                mMediaRecorder.setAudioSamplingRate(44100);
                mMediaRecorder.setAudioEncodingBitRate(16 * 44100);
                mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
                mMediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
                mMediaRecorder.setOutputFile(saveFile);
                mMediaRecorder.prepare();
                mMediaRecorder.start();
            } catch (IOException e) {
                e.printStackTrace();
            }
            scheduledExecutorService.schedule(() -> {
                try {
                    if (mMediaRecorder != null) {
                        mMediaRecorder.stop();
                        mMediaRecorder.release();
                        mMediaRecorder = null;
                    }
                    isRecording.set(false);
                    ft.complete(null);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }, config.getAudioLength(), TimeUnit.MILLISECONDS);
        } else {
            ft.complete(null);
        }
        return ft;
    }
     */

    @Override
    public void close() {
        stopRecording();
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        CollectorResult result = new CollectorResult();
        File saveFile = new File(config.getAudioFilename());
        result.setSavePath(saveFile.getAbsolutePath());

        if (config.getAudioLength() <= 0) {
            result.setErrorCode(1);
            result.setErrorReason("Invalid audio length: " + config.getAudioLength());
//            ft.complete(result);
            ft.completeExceptionally(new Exception("Invalid audio length: " + config.getAudioLength()));
        } else if (config.getAudioFilename() == null) {
            result.setErrorCode(2);
            result.setErrorReason("Null audio filename");
//            ft.complete(result);
            ft.completeExceptionally(new Exception("Null audio filename"));
        } else if (isCollecting.compareAndSet(false, true)) {
            try {
                FileUtils.makeDir(saveFile.getParent());
                startRecording(saveFile);
                futureList.add(scheduledExecutorService.schedule(() -> {
                    try {
                        stopRecording();
                        ft.complete(result);
                    } catch (Exception e) {
                        e.printStackTrace();
                        result.setErrorCode(5);
                        result.setErrorReason(e.toString());
                        ft.completeExceptionally(e);
                    } finally {
//                        ft.complete(result);
                        isCollecting.set(false);
                    }
                }, config.getAudioLength(), TimeUnit.MILLISECONDS));
            } catch (Exception e) {
                e.printStackTrace();
                stopRecording();
                result.setErrorCode(4);
                result.setErrorReason(e.toString());
//                ft.complete(result);
                ft.completeExceptionally(e);
                isCollecting.set(false);
            }
        } else {
            result.setErrorCode(3);
            result.setErrorReason("Concurrent task of audio recording");
//            ft.complete(result);
            ft.completeExceptionally(new Exception("Concurrent task of audio recording"));
        }

        return ft;
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    private void startRecording(File file) throws IOException {
        mMediaRecorder = new MediaRecorder();
        mMediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mMediaRecorder.setAudioChannels(2);
        mMediaRecorder.setAudioSamplingRate(44100);
        mMediaRecorder.setAudioEncodingBitRate(16 * 44100);
        mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mMediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        mMediaRecorder.setOutputFile(file);
        mMediaRecorder.prepare();
        mMediaRecorder.start();
    }

    private void stopRecording() {
        if (mMediaRecorder != null) {
            mMediaRecorder.stop();
            mMediaRecorder.release();
            mMediaRecorder = null;
        }
    }

    /*
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return null;
    }

     */

    @Override
    public void pause() {
        stopRecording();
    }

    @Override
    public void resume() {

    }

    @Override
    public String getName() {
        return "Audio";
    }

    @Override
    public String getExt() {
        return ".mp3";
    }
}
