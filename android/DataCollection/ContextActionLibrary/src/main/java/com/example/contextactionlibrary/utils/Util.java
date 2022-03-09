package com.example.contextactionlibrary.utils;

import java.util.ArrayList;

public class Util {
    public static int getMaxId(ArrayList<Float> var0) {
        float var1 = -3.4028235E38F;
        int var2 = 0;

        int var3;
        for(var3 = 0; var2 < var0.size(); ++var2) {
            if (var1 < (Float)var0.get(var2)) {
                var1 = (Float)var0.get(var2);
                var3 = var2;
            }
        }

        return var3;
    }
}
