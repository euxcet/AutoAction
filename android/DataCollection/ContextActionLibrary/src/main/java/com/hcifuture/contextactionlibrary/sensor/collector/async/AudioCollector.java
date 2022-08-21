package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;
import android.media.AudioManager;
import android.media.MediaRecorder;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorException;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.contextactionlibrary.utils.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
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
        2: Null audio filename
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

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        Heart.getInstance().newSensorGetEvent(getName(), System.currentTimeMillis());
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        CollectorResult result = new CollectorResult();
        File saveFile = new File(config.getAudioFilename());
        result.setSavePath(saveFile.getAbsolutePath());

        if (config.getAudioLength() <= 0) {
            ft.completeExceptionally(new CollectorException(1, "Invalid audio length: " + config.getAudioLength()));
        } else if (config.getAudioFilename() == null) {
            ft.completeExceptionally(new CollectorException(2, "Null audio filename"));
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    private CompletableFuture<List<Integer>> getMaxAmplitudeSequence(long length, long period) {
        CompletableFuture<List<Integer>> ft = new CompletableFuture<>();
        if (isCollecting.compareAndSet(false, true)) {
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
                    try {
                        long start_time = System.currentTimeMillis();

//                        final MediaRecorder mediaRecorder_comm = startNewMediaRecorder(MediaRecorder.AudioSource.VOICE_COMMUNICATION, getDummyOutputFilePath()+"comm");
//                        Log.e(TAG, String.format("getMaxAmplitudeSequence: MediaRecorder comm started, length: %dms period: %dms", length, period));
//                        // first call returns 0
//                        mediaRecorder_comm.getMaxAmplitude();

                        mMediaRecorder = startNewMediaRecorder(MediaRecorder.AudioSource.MIC, getDummyOutputFilePath()+"mic");
                        Log.e(TAG, String.format("getMaxAmplitudeSequence: MediaRecorder mic started, length: %dms period: %dms", length, period));
                        // first call returns 0
                        mMediaRecorder.getMaxAmplitude();

                        List<Integer> sampledNoise_mic = new ArrayList<>();
//                        List<Integer> sampledNoise_comm = new ArrayList<>();
                        repeatedSampleFt = scheduledExecutorService.scheduleAtFixedRate(() -> {
                            try {
                                // Returns the maximum absolute amplitude that was sampled since the last call to this method
                                int maxAmplitude = mMediaRecorder.getMaxAmplitude();
                                sampledNoise_mic.add(maxAmplitude);
//                                int maxAmplitude_comm = mediaRecorder_comm.getMaxAmplitude();
//                                sampledNoise_comm.add(maxAmplitude_comm);
                                if (System.currentTimeMillis() - start_time + period > length) {
                                    stopMediaRecorder(mMediaRecorder);
//                                    stopMediaRecorder(mediaRecorder_comm);

                                    Log.e(TAG, "getMaxAmplitudeSequence: mic  " + sampledNoise_mic);
//                                    Log.e(TAG, "getMaxAmplitudeSequence: comm " + sampledNoise_comm);

//                                    sampledNoise_mic.addAll(sampledNoise_comm);
                                    repeatedSampleFt.cancel(false);
                                    isCollecting.set(false);
                                    ft.complete(sampledNoise_mic);
                                }
                            } catch (Exception e) {
                                e.printStackTrace();
                                stopMediaRecorder(mMediaRecorder);
                                repeatedSampleFt.cancel(false);
                                isCollecting.set(false);
                                ft.completeExceptionally(new CollectorException(5, e));
                            }
                        }, period, period, TimeUnit.MILLISECONDS);
                        futureList.add(repeatedSampleFt);

                    } catch (Exception e) {
                        stopMediaRecorder(mMediaRecorder);
                        isCollecting.set(false);
                        ft.completeExceptionally(new CollectorException(7, e));
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                isCollecting.set(false);
                ft.completeExceptionally(new CollectorException(4, e));
            }
        } else {
            ft.completeExceptionally(new CollectorException(3, "Concurrent task of audio recording"));
        }

        return ft;
    }

    private double getAvgNoiseFromSeq(List<Integer> seq) {
        double BASE = 1.0;
        double sum = 0.0;
        int count = 0;

        int idx = 0;
        double db;
        int next_idx;
        double next_db;
        int maxAmplitude = 0;

        // 找到第一个非零值
        while (idx < seq.size() && (maxAmplitude = seq.get(idx)) == 0) {
            idx++;
        }
        if (idx >= seq.size()) {
            // 没有非零值
            throw new CollectorException(8, "No MaxAmplitude > 0");
        }
        db = 20 * Math.log10(maxAmplitude / BASE);
        next_idx = idx + 1;
//            Log.e(TAG, "getNoiseLevel: " + String.format("idx: %d maxAmplitude: %d db: %f", idx, maxAmplitude, db));
        // 采样为0时使用两边非零值线性插值
        while (true) {
            while (next_idx < seq.size() && (maxAmplitude = seq.get(next_idx)) == 0) {
                next_idx++;
            }
            if (next_idx >= seq.size()) {
                sum += db;
                count += 1;
                break;
            }
            next_db = 20 * Math.log10(maxAmplitude / BASE);
            sum += db + (db + next_db) * 0.5 * (next_idx - idx - 1);
            count += next_idx - idx;

            idx = next_idx++;
            db = next_db;

//                Log.e(TAG, "getNoiseLevel: " + String.format("idx: %d maxAmplitude: %d db: %f", idx, maxAmplitude, db));
        }

        double average_noise = (count > 0)? (sum / count) : 0.0;

        Log.e(TAG, String.format("getNoiseLevel: %d sampled, average %fdb", count, average_noise));
        return average_noise;
    }

    // get current noise level
    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<Double> getNoiseLevel(long length, long period) {
        return getMaxAmplitudeSequence(length, period).thenApply(seq -> {
//            int num = combinedSeq.size() / 2;
//            List<Integer> seq_mic = combinedSeq.subList(0, num);
//            List<Integer> seq_comm = combinedSeq.subList(num, combinedSeq.size());
//            Log.e(TAG, "getNoiseLevel: seq number " + num + " combined: " + combinedSeq.size());

//            double db_mic = getAvgNoiseFromSeq(seq_mic);
//            double db_comm = getAvgNoiseFromSeq(seq_comm);
//            return Arrays.asList(db_mic, db_comm);

            return getAvgNoiseFromSeq(seq);
        });
    }
}
