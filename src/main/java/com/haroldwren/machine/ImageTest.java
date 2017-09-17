package com.haroldwren.machine;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.train.strategy.ResetStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.platformspecific.j2se.data.image.ImageMLData;
import org.encog.platformspecific.j2se.data.image.ImageMLDataSet;
import org.encog.util.downsample.Downsample;
import org.encog.util.downsample.RGBDownsample;
import org.encog.util.simple.EncogUtility;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ImageTest {
    private final List<ImagePair> imageList = new ArrayList<>();
    private final Map<String, Integer> identity2neuron = new HashMap<>();
    private final Map<Integer, String> neuron2identity = new HashMap<>();
    private ImageMLDataSet training;
    private int outputCount;
    private int downsampleWidth = 64;
    private int downsampleHeight = 64;
    private BasicNetwork network;

    private Downsample downsample  = new RGBDownsample(); // new SimpleIntensityDownsample();

    public static void main(final String[] args) throws IOException, URISyntaxException {
        ImageTest imageTest = new ImageTest();
        imageTest.processCreateTraining();

        File folder = new File(ImageTest.class.getResource("/betanitas").toURI());

        imageTest.processInput(folder.getPath()+"/cat1.jpeg", "cat");
        imageTest.processInput(folder.getPath()+"/cat2.jpeg", "cat");
        imageTest.processInput(folder.getPath()+"/cat3.jpg", "cat");
//        imageTest.processInput(folder.getPath()+"/cat4.jpg", "cat");
        imageTest.processInput(folder.getPath()+"/cat5.jpg", "cat");
        imageTest.processInput(folder.getPath()+"/cat6.jpg", "cat");
        imageTest.processInput(folder.getPath()+"/cat7.jpg", "cat");
        imageTest.processInput(folder.getPath()+"/house1.jpeg", "not_cat");
        imageTest.processInput(folder.getPath()+"/house2.jpeg", "not_cat");
        imageTest.processInput(folder.getPath()+"/house3.jpeg", "not_cat");
        imageTest.processInput(folder.getPath()+"/house4.jpg", "not_cat");
//        imageTest.processInput(folder.getPath()+"/house5.jpg", "not_cat");
        imageTest.processInput(folder.getPath()+"/house6.jpg", "not_cat");

        imageTest.processNetwork(100, 0);

        imageTest.processTrain( 1, 0.25, 50);

        imageTest.processWhatIs(folder.getPath()+"/cat4.jpg");
        imageTest.processWhatIs(folder.getPath()+"/house5.jpg");
    }

    private void processCreateTraining() {
        this.training = new ImageMLDataSet(this.downsample, false, 1, -1);
        System.out.println("Training set created");
    }

    private int assignIdentity(final String identity) {
        if (this.identity2neuron.containsKey(identity.toLowerCase())) {
            return this.identity2neuron.get(identity.toLowerCase());
        }

        final int result = this.outputCount;
        this.identity2neuron.put(identity.toLowerCase(), result);
        this.neuron2identity.put(result, identity.toLowerCase());
        this.outputCount++;
        return result;
    }

    private void processInput(final String image, final String identity) throws IOException {
        final int idx = assignIdentity(identity);
        final File file = new File(image);

        this.imageList.add(new ImagePair(file, idx));

        System.out.println("Added input image:" + image);
    }

    private void processNetwork(final int hidden1, final int hidden2) throws IOException {
        System.out.println("Downsampling images...");

        for (final ImagePair pair : this.imageList) {
            final MLData ideal = new BasicMLData(this.outputCount);
            final int idx = pair.getIdentity();
            for (int i = 0; i < this.outputCount; i++) {
                if (i == idx) {
                    ideal.setData(i, 1);
                } else {
                    ideal.setData(i, -1);
                }
            }

            final Image img = ImageIO.read(pair.getFile());
            final ImageMLData data = new ImageMLData(img);
            this.training.add(data, ideal);
        }

        this.training.downsample(this.downsampleHeight, this.downsampleWidth);

        this.network = EncogUtility.simpleFeedForward(this.training
                        .getInputSize(), hidden1, hidden2,
                this.training.getIdealSize(), true);
        System.out.println("Created network: " + this.network.toString());
    }

    private void processTrain(final int minutes, final double strategyError, final int strategyCycles)
                    throws IOException {
        System.out.println("Training Beginning... Output patterns="
                + this.outputCount);

        final ResilientPropagation train = new ResilientPropagation(this.network, this.training);
        train.addStrategy(new ResetStrategy(strategyError, strategyCycles));

        EncogUtility.trainConsole(train, this.network, this.training,
                minutes);
        System.out.println("Training Stopped...");
    }

    public void processWhatIs(final String filename) throws IOException {
        final File file = new File(filename);
        final Image img = ImageIO.read(file);
        final ImageMLData input = new ImageMLData(img);
        input.downsample(this.downsample, false, this.downsampleHeight,
                this.downsampleWidth, 1, -1);
        final int winner = this.network.winner(input);
        System.out.println("What is: " + filename + ", it seems to be: "
                + this.neuron2identity.get(winner));
    }

}
