package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;
import android.media.MediaRecorder;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

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
    private File saveFile;
    private AtomicBoolean isRecording;

    public AudioCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        isRecording = new AtomicBoolean(false);
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

    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        if (config.getAudioLength() <= 0) {
            ft.completeExceptionally(new Exception("Invalid audio length: " + config.getAudioLength()));
            return ft;
        }
        if (config.getAudioFilename() == null) {
            ft.completeExceptionally(new Exception("NULL audio filename!"));
            return ft;
        }
        saveFile = new File(config.getAudioFilename());
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
                mMediaRecorder.setOutputFile(new File(config.getAudioFilename()));
                mMediaRecorder.prepare();
                mMediaRecorder.start();
                scheduledExecutorService.schedule(() -> {
                    try {
                        if (mMediaRecorder != null) {
                            mMediaRecorder.stop();
                            mMediaRecorder.release();
                            mMediaRecorder = null;
                        }
                        isRecording.set(false);
                        ft.complete(new CollectorResult().setSavePath(saveFile.getAbsolutePath()));
                    } catch (Exception e) {
                        e.printStackTrace();
                        ft.completeExceptionally(e);
                    }
                }, config.getAudioLength(), TimeUnit.MILLISECONDS);
            } catch (IOException e) {
                e.printStackTrace();
                ft.completeExceptionally(e);
            }
        } else {
            ft.completeExceptionally(new Exception("Another task of audio recording is taking place!"));
        }
        return ft;
    }

    /*
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return null;
    }

     */

    @Override
    public void pause() {

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
