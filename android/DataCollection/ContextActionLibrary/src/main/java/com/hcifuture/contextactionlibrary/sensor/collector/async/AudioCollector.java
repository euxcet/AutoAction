package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;
import android.media.AudioManager;
import android.media.MediaRecorder;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorException;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.contextactionlibrary.utils.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class AudioCollector extends AsynchronousCollector {
    private static final String TAG = "AudioCollector";

    private MediaRecorder mMediaRecorder;
    private final AtomicBoolean isCollecting;
    private final AudioManager audioManager;

    private ScheduledFuture<?> repeatedSampleFt;
    private final File dummyOutputFile;

    /*
      Error code:
        0: No error
        1: Invalid audio length
        2: Null audio filename (deprecated)
        3: Concurrent task of audio recording
        4: Unknown audio recording exception
        5: Unknown exception when stopping recording
        6: Mic not available
        7: Exception during getMaxAmplitudeSequence (noise detection)
        8: No valid result of getMaxAmplitudeSequence (noise value)
     */

    public AudioCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        isCollecting = new AtomicBoolean(false);
        audioManager = (AudioManager) mContext.getSystemService(Context.AUDIO_SERVICE);

        dummyOutputFile = new File(context.getExternalMediaDirs()[0].getAbsolutePath() + "/tmp/null");
        FileUtils.makeDir(dummyOutputFile.getParent());
    }

    @Override
    public void initialize() {

    }

    @Override
    public void close() {
        stopRecording();
        clearDummyOutputFile();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        Heart.getInstance().newSensorGetEvent(getName(), System.currentTimeMillis());
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        CollectorResult result = new CollectorResult();
        String filename;
        // if audio filename not specified, use dummy output file
        if (config.getAudioFilename() == null) {
            filename = getDummyOutputFilePath();
        } else {
            filename = config.getAudioFilename();
        }
        File saveFile = new File(filename);
        result.setSavePath(saveFile.getAbsolutePath());

        if (config.getAudioLength() <= 0) {
            ft.completeExceptionally(new CollectorException(1, "Invalid audio length: " + config.getAudioLength()));
//        } else if (config.getAudioFilename() == null) {
//            ft.completeExceptionally(new CollectorException(2, "Null audio filename"));
        } else if (isCollecting.compareAndSet(false, true)) {
            try {
                // check mic availability
                // ref: https://stackoverflow.com/a/67458025/11854304
//                MODE_NORMAL -> You good to go. Mic not in use
//                MODE_RINGTONE -> Incoming call. The phone is ringing
//                MODE_IN_CALL -> A phone call is in progress
//                MODE_IN_COMMUNICATION -> The Mic is being used by another application
                int micMode = audioManager.getMode();
                if (micMode != AudioManager.MODE_NORMAL) {
                    isCollecting.set(false);
                    ft.completeExceptionally(new CollectorException(6, "Mic not available: " + micMode));
                } else {
                    FileUtils.makeDir(saveFile.getParent());
                    startRecording(saveFile);
                    // first call returns 0
                    mMediaRecorder.getMaxAmplitude();
                    futureList.add(scheduledExecutorService.schedule(() -> {
                        try {
                            stopRecording();
                            isCollecting.set(false);
                            ft.complete(result);
                        } catch (Exception e) {
                            e.printStackTrace();
                            isCollecting.set(false);
                            ft.completeExceptionally(new CollectorException(5, e));
                        }
                    }, config.getAudioLength(), TimeUnit.MILLISECONDS));
                }
            } catch (Exception e) {
                e.printStackTrace();
                stopRecording();
                // remove file
                FileUtils.deleteFile(saveFile, "");
                isCollecting.set(false);
                ft.completeExceptionally(new CollectorException(4, e));
            }
        } else {
            ft.completeExceptionally(new CollectorException(3, "Concurrent task of audio recording"));
        }

        return ft;
    }

    private void startRecording(File file) throws IOException {
        mMediaRecorder = startNewMediaRecorder(MediaRecorder.AudioSource.MIC, file.getAbsolutePath());
    }

    private void stopRecording() {
        stopMediaRecorder(mMediaRecorder);
    }

    @Override
    public void pause() {
        stopRecording();
        clearDummyOutputFile();
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

    public String getDummyOutputFilePath() {
        // from Android 11 (SDK 30) on, cannot use "/dev/null"
        return dummyOutputFile.getAbsolutePath();
    }

    public void clearDummyOutputFile() {
        try {
            FileUtils.deleteFile(dummyOutputFile, "");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private MediaRecorder startNewMediaRecorder(int audioSource, String outputFilePath) throws IOException {
        MediaRecorder mediaRecorder = new MediaRecorder();
        // may throw IllegalStateException due to lack of permission
        mediaRecorder.setAudioSource(audioSource);
        mediaRecorder.setAudioChannels(2);
        mediaRecorder.setAudioSamplingRate(44100);
        mediaRecorder.setAudioEncodingBitRate(16 * 44100);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        mediaRecorder.setOutputFile(outputFilePath);
        mediaRecorder.prepare();
        mediaRecorder.start();
        return mediaRecorder;
    }

    private void stopMediaRecorder(MediaRecorder mediaRecorder) {
        if (mediaRecorder != null) {
            try {
                // may throw IllegalStateException because no valid audio data has been received
                mediaRecorder.stop();
            } catch (Exception e) {
                e.printStackTrace();
            }
            try {
                mediaRecorder.release();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public int getMaxAmplitude() {
        try {
            // Returns the maximum absolute amplitude that was sampled since the last call to this method
            return mMediaRecorder.getMaxAmplitude();
        } catch (Exception e) {
            return -1;
        }
    }
}
