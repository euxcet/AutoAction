package com.hcifuture.contextactionlibrary.contextaction.action.tapfilter;

public class CombinedFilter extends Filter {
    private boolean existTaptapSignal = false;
    private long signalTime;
    private long confirmedTime;
    private long lastAvoidedTime;

    @Override
    public int passWithDelay(long timestamp) {
        if (existTaptapSignal) {
            if (timestamp - signalTime > 500 * MILLISECOND) { // avoid
                existTaptapSignal = false;
                signalTime = 0;
                confirmedTime = 0;
                lastAvoidedTime = timestamp;
                return -1;
            }
            if (Math.abs(confirmedTime - signalTime) < 500 * MILLISECOND || timestamp - lastAvoidedTime < 5 * SECOND) { // pass
//                Log.e("TapTap", "Interval " + String.valueOf((int)((confirmedTime - signalTime) / MILLISECOND)));
                existTaptapSignal = false;
                signalTime = 0;
                confirmedTime = 0;
                lastAvoidedTime = 0;
                return 1;
            }
        }
        return 0;
    }

    @Override
    public boolean passDirectly() {
        return false;
    }

    public void updateCondition() {
        existTaptapSignal = true;
        signalTime = lastTime[1];
    }

    public void confirmed() {
        confirmedTime = lastTime[1];
    }
}
