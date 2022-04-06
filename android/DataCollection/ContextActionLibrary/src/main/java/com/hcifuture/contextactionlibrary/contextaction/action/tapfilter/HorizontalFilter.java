package com.hcifuture.contextactionlibrary.contextaction.action.tapfilter;

public class HorizontalFilter extends Filter {
    private boolean existTaptapSignal = false;
    private long signalTime;
    private long lastAvoidedTime;

    // just for horizontal / static cases' record && upload
    @Override
    public int passWithDelay(long timestamp) {
        if (existTaptapSignal) {
            if (timestamp - signalTime < 3 * SECOND) {
                if (linearStaticCount > 50 && gyroStaticCount > 100) {  // static
                    existTaptapSignal = false;
                    signalTime = 0;
                    lastAvoidedTime = timestamp;
                    return -1;
                }
            }
            else {  // pass
                existTaptapSignal = false;
                signalTime = 0;
                lastAvoidedTime = 0;
                return 1;
            }
        }
        return 0;
    }

//    @Override
//    public int passWithDelay(long timestamp) {
//        if (existTaptapSignal) {
//            int yAllowed = timestamp - signalTime < 150 * MILLISECOND ? 4 : 2;
//            int zAllowed = timestamp - signalTime < 150 * MILLISECOND ? 3 : 2;
//
//            if ((checkIsHorizontal(yAllowed, zAllowed) || timestamp - signalTime < 50 * MILLISECOND) && timestamp - signalTime < SECOND && timestamp - lastAvoidedTime > 5 * SECOND) {
//                if (checkIsStatic()) { // avoid
//                    existTaptapSignal = false;
//                    signalTime = 0;
//                    lastAvoidedTime = timestamp;
//                    return -1;
//                }
//            }
//            else { // pass
//                existTaptapSignal = false;
//                signalTime = 0;
//                lastAvoidedTime = 0;
//                return 1;
//            }
//        }
//        return 0;
//    }

    @Override
    public boolean passDirectly() {
        if (!checkIsHorizontal(5, 5)) {
            return true;
        } else {
            existTaptapSignal = true;
            signalTime = lastTime[1];
            return false;
        }
    }

    public void updateCondition() {
        existTaptapSignal = true;
        signalTime = lastTime[1];
    }
}
