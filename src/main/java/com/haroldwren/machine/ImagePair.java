package com.haroldwren.machine;

import java.io.File;

public class ImagePair {
    private final File file;
    private final int identity;

    public ImagePair(final File file, final int identity) {
        super();
        this.file = file;
        this.identity = identity;
    }

    public File getFile() {
        return this.file;
    }

    public int getIdentity() {
        return this.identity;
    }
}
