package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.Context;
import android.media.MediaRecorder;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;
import com.hcifuture.contextactionlibrary.collect.trigger.TriggerConfig;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class AudioCollector extends Collector {
    private MediaRecorder mMediaRecorder;
    private File saveFile;
    private AtomicBoolean isRecording;

    public AudioCollector(Context context, Trigger.CollectorType type, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, triggerFolder, scheduledExecutorService, futureList);
        isRecording = new AtomicBoolean(false);
    }

    @Override
    public void initialize() {

    }

    @Override
    public void setSavePath(String timestamp) {
        saver.setSavePath(timestamp + "_audio.mp3");
        saveFile = new File(saver.getSavePath());
    }

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

    @Override
    public void close() {

    }

    @Override
    public boolean forPrediction() {
        return false;
    }

    @Override
    public Data getData() {
        return null;
    }

    @Override
    public String getSaveFolderName() {
        return "Audio";
    }

    @Override
    public void pause() {

    }

    @Override
    public void resume() {

    }
}
