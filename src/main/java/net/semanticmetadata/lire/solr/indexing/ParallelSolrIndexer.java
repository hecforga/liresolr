/*
 * This file is part of the LIRE project: http://www.semanticmetadata.net/lire
 * LIRE is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LIRE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with LIRE; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * We kindly ask you to refer the any or one of the following publications in
 * any publication mentioning or employing Lire:
 *
 * Lux Mathias, Savvas A. Chatzichristofis. Lire: Lucene Image Retrieval –
 * An Extensible Java CBIR Library. In proceedings of the 16th ACM International
 * Conference on Multimedia, pp. 1085-1088, Vancouver, Canada, 2008
 * URL: http://doi.acm.org/10.1145/1459359.1459577
 *
 * Lux Mathias. Content Based Image Retrieval with LIRE. In proceedings of the
 * 19th ACM International Conference on Multimedia, pp. 735-738, Scottsdale,
 * Arizona, USA, 2011
 * URL: http://dl.acm.org/citation.cfm?id=2072432
 *
 * Mathias Lux, Oge Marques. Visual Information Retrieval using Java and LIRE
 * Morgan & Claypool, 2013
 * URL: http://www.morganclaypool.com/doi/abs/10.2200/S00468ED1V01Y201301ICR025
 *
 * Copyright statement:
 * --------------------
 * (c) 2002-2013 by Mathias Lux (mathias@juggle.at)
 *     http://www.semanticmetadata.net/lire, http://www.lire-project.net
 */

package net.semanticmetadata.lire.solr.indexing;

import net.semanticmetadata.lire.imageanalysis.features.GlobalFeature;
import net.semanticmetadata.lire.imageanalysis.features.global.*;
import net.semanticmetadata.lire.indexers.hashing.BitSampling;
import net.semanticmetadata.lire.indexers.hashing.MetricSpaces;
import net.semanticmetadata.lire.indexers.parallel.WorkItem;
import net.semanticmetadata.lire.solr.FeatureRegistry;
import net.semanticmetadata.lire.solr.HashingMetricSpacesManager;
import net.semanticmetadata.lire.utils.ImageUtils;

import javax.imageio.ImageIO;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.request.DirectXmlRequest;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.impl.HttpSolrClient;
import org.apache.solr.common.util.ContentStreamBase;
import org.w3c.dom.Document;


/**
 * This indexing application allows for parallel extraction of global features from multiple image files for
 * use with the LIRE Solr plugin. It basically takes a list of images (ie. created by something like
 * "dir /s /b &gt; list.txt" or "ls [some parameters] &gt; list.txt".
 *
 * use it like:
 * <pre>$&gt; java -jar lire-request-handler.jar -i &lt;infile&gt; [-o &lt;outfile&gt;] [-n &lt;threads&gt;] [-m &lt;max_side_length&gt;] [-f]</pre>
 *
 * Available options are:
 * <ul>
 * <li> -i &lt;infile&gt; ... gives a file with a list of images to be indexed, one per line.</li>
 * <li> -o &lt;outfile&gt; ... gives XML file the output is written to. if none is given the outfile is &lt;infile&gt;.xml</li>
 * <li> -n &lt;threads&gt; ... gives the number of threads used for extraction. The number of cores is a good value for that.</li>
 * <li> -m &lt;max-side-length&gt; ... gives a maximum side length for extraction. This option is useful if very larger images are indexed.</li>
 * <li> -f ... forces to overwrite the &lt;outfile&gt;. If the &lt;outfile&gt; already exists and -f is not given, then the operation is aborted.</li>
 * <li> -p ... enables image processing before indexing (despeckle, trim white space)</li>
 * <li> -a ... use both BitSampling and MetricSpaces.</li>
 * <li> -l ... disables BitSampling and uses MetricSpaces instead.</li>
 * <li> -r ... defines a class implementing net.semanticmetadata.lire.solr.indexing.ImageDataProcessor that provides additional fields.</li>
 * </ul>
 * <p>
 * TODO: Make feature list change-able
 * </p>
 * You then basically need to enrich the file with whatever metadata you prefer and send it to Solr using for instance curl:
 * <pre>curl http://localhost:9000/solr/lire/update  -H "Content-Type: text/xml" --data-binary @extracted_file.xml
 * curl http://localhost:9000/solr/lire/update  -H "Content-Type: text/xml" --data-binary "&lt;commit/&gt;"</pre>
 *
 * @author Mathias Lux, mathias@juggle.at on  13.08.2013
 */
public class ParallelSolrIndexer implements Runnable {

    private static ArrayList<String> genderList = new ArrayList<String>();
    private static ArrayList<String> categoryList = new ArrayList<String>();

    private final int maxCacheSize = 250;
    //    private static HashMap<Class, String> classToPrefix = new HashMap<Class, String>(5);
    private boolean force = false;
    private static boolean individualFiles = false;
    private static int numberOfThreads = 8;

    private boolean useMetricSpaces = false, useBitSampling = true;

    LinkedBlockingQueue<WorkItem> images = new LinkedBlockingQueue<WorkItem>(maxCacheSize);
    boolean ended = false;
    int overallCount = 0;
    OutputStream dos = null;
    Set<Class> listOfFeatures;

    ArrayList<File> fileList;
    File outFile = new File("outfile.xml");
    private int monitoringInterval = 10;
    private int maxSideLength = 512;
    private boolean isPreprocessing = true;
    private Class imageDataProcessor = null;


    public ParallelSolrIndexer() {
        // default constructor.
        fileList = new ArrayList<File>();
        listOfFeatures = new HashSet<Class>();
        listOfFeatures.add(CEDD.class);
        listOfFeatures.add(FCTH.class);
        listOfFeatures.add(JCD.class);
        listOfFeatures.add(AutoColorCorrelogram.class);
        /*listOfFeatures.add(ScalableColor.class);
        listOfFeatures.add(ColorLayout.class);
        listOfFeatures.add(EdgeHistogram.class);
        listOfFeatures.add(Tamura.class);
        listOfFeatures.add(Gabor.class);
        listOfFeatures.add(SimpleColorHistogram.class);
        listOfFeatures.add(OpponentHistogram.class);
        listOfFeatures.add(JointHistogram.class);
        listOfFeatures.add(LuminanceLayout.class);
        listOfFeatures.add(PHOG.class);
        listOfFeatures.add(ACCID.class);*/
        HashingMetricSpacesManager.init(); // load reference points from disk.

    }

    public static void main(String[] args) throws IOException {
        BitSampling.readHashFunctions();

        // parse programs args ...
        if (args.length < 3) {
            System.err.println("Wrong number of arguments. 3 needed. Usage example:\n" +
                    "home/datsetDirectory mujer all");
        }
        else{
            String genders = args[1];
            String categories = args[2];

            if(genders.equals("all")){

                genderList.add("mujer");
                genderList.add("hombre");
            }else {
                genderList.add(genders);
            }
            if(categories.equals("all")){
                categoryList.add("abrigos_chaquetas");
                categoryList.add("camisas_blusas");
                categoryList.add("camisetas_tops_bodies");
                categoryList.add("faldas");
                categoryList.add("monos");
                categoryList.add("pantalones_cortos");
                categoryList.add("pantalones_largos");
                categoryList.add("punto");
                categoryList.add("sudaderas_jerseis");
                categoryList.add("vestidos");
            }else {
                categoryList.add(categories);
            }
        }
        String pathToDataset = args[0];
        for (String gender:genderList) {
            for (String category:categoryList) {
                //setFileList
                ParallelSolrIndexer e = new ParallelSolrIndexer();
                System.out.println("Empezando con: "+category);
                File croppedFolder = new File(pathToDataset+"/"+gender+"/"+category+"/CROPPED");
                File [] croppedFileNames = croppedFolder.listFiles();
                ArrayList<File> listOfFiles = new ArrayList<File>(Arrays.asList(croppedFileNames));
                e.appendFileList(listOfFiles);
                e.setForce(true);
                try {
                    ParallelSolrIndexer.numberOfThreads = 8;
                } catch (Exception e1) {
                    System.err.println("Could not set number of threads to 8.");
                    e1.printStackTrace();
                }
                e.run();
                e.postIndexToServer(gender, category);
                e.resetGlobalVariables();
            }
        }
    }

    public static String arrayToString(int[] array) {
        StringBuilder sb = new StringBuilder(array.length * 8);
        for (int i = 0; i < array.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(Integer.toHexString(array[i]));
        }
        return sb.toString();
    }

    public void appendFileList(ArrayList<File> fileList) {
        this.fileList.addAll(fileList);
    }

    public void resetGlobalVariables() {
        fileList.clear();
    }

    public void postIndexToServer(String gender, String category) {
        try {
            SolrClient client = new HttpSolrClient.Builder("http://139.59.155.103:8983/solr/"+gender+"_"+category).build();

            DocumentBuilderFactory dbfac = DocumentBuilderFactory.newInstance();
            DocumentBuilder docBuilder = dbfac.newDocumentBuilder();
            Document doc = docBuilder.parse(outFile);
            TransformerFactory tf = TransformerFactory.newInstance();
            Transformer transformer = tf.newTransformer();
            transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
            StringWriter writer = new StringWriter();
            transformer.transform(new DOMSource(doc), new StreamResult(writer));
            String xmlOutput = writer.getBuffer().toString().replaceAll("\n|\r", "");

            DirectXmlRequest xmlreq = new DirectXmlRequest( "/update", xmlOutput);
            client.deleteByQuery("*:*");
            //client.request(xmlreq);
            client.commit();
        }catch (SolrServerException se){
            System.err.println("Caught SolrServerException: " + se.getMessage());
        }catch (IOException e){
            System.err.println("Caught IOException: " + e.getMessage());
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    @Override
    public void run() {
        // check:
        if (fileList == null) {
            System.err.println("No text file with a list of images given.");
            return;
        }
        System.out.println("Extracting features: ");
        for (Iterator<Class> iterator = listOfFeatures.iterator(); iterator.hasNext(); ) {
            System.out.println("\t" + iterator.next().getCanonicalName());
        }
        try {
            if (!individualFiles) {
                // create a BufferedOutputStream with a large buffer
                dos = new BufferedOutputStream(new FileOutputStream(outFile), 1024 * 1024 * 8);
                dos.write("<add>\n".getBytes());
            }
            Thread p = new Thread(new Producer(), "Producer");
            p.start();
            LinkedList<Thread> threads = new LinkedList<Thread>();
            long l = System.currentTimeMillis();
            for (int i = 0; i < numberOfThreads; i++) {
                Thread c = new Thread(new Consumer(), "Consumer-" + i);
                c.start();
                threads.add(c);
            }
            Thread m = new Thread(new Monitoring(), "Monitoring");
            m.start();
            for (Iterator<Thread> iterator = threads.iterator(); iterator.hasNext(); ) {
                iterator.next().join();
            }
            long l1 = System.currentTimeMillis() - l;
            System.out.println("Analyzed " + overallCount + " images in " + l1 / 1000 + " seconds, ~" + (overallCount > 0 ? (l1 / overallCount) : "inf.") + " ms each.");
            if (!individualFiles) {
                dos.write("</add>\n".getBytes());
                dos.close();
            }
//            writer.commit();
//            writer.close();
//            threadFinished = true;

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void addFeatures(List features) {
        for (Iterator<Class> iterator = listOfFeatures.iterator(); iterator.hasNext(); ) {
            Class next = iterator.next();
            try {
                features.add(next.newInstance());
            } catch (InstantiationException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }
    }

    public void setUseMetricSpaces(boolean useMetricSpaces) {
        this.useMetricSpaces = useMetricSpaces;
        this.useBitSampling = !useMetricSpaces;
    }

    public boolean isPreprocessing() {
        return isPreprocessing;
    }

    public void setPreprocessing(boolean isPreprocessing) {
        this.isPreprocessing = isPreprocessing;
    }

    public boolean isForce() {
        return force;
    }

    public void setForce(boolean force) {
        this.force = force;
    }

    public void setUseBothHashingAlgortihms(boolean useBothHashingAlgortihms) {
        this.useMetricSpaces = useBothHashingAlgortihms;
        this.useBitSampling = useBothHashingAlgortihms;
    }

    class Monitoring implements Runnable {
        public void run() {
            long ms = System.currentTimeMillis();
            try {
                Thread.sleep(1000 * monitoringInterval); // wait xx seconds
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            while (!ended) {
                try {
                    // print the current status:
                    long time = System.currentTimeMillis() - ms;
                    System.out.println("Analyzed " + overallCount + " images in " + time / 1000 + " seconds, " + ((overallCount > 0) ? (time / overallCount) : "n.a.") + " ms each (" + images.size() + " images currently in queue).");
                    Thread.sleep(1000 * monitoringInterval); // wait xx seconds
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    class Producer implements Runnable {
        public void run() {

            for(File next: fileList) {
                try {
                    // reading from harddrive to buffer to reduce the load on the HDD and move decoding to the
                    // consumers using java.nio
                    int fileSize = (int) next.length();
                    byte[] buffer = new byte[fileSize];
                    FileInputStream fis = new FileInputStream(next);
                    FileChannel channel = fis.getChannel();
                    MappedByteBuffer map = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
                    map.load();
                    map.get(buffer);
                    String path = next.getCanonicalPath();
                    images.put(new WorkItem(path, buffer));
                } catch (Exception e) {
                    System.err.println("Could not read image " + next.getName() + ": " + e.getMessage());
                }
            }
            for (int i = 0; i < numberOfThreads*2; i++) {
                String tmpString = null;
                byte[] tmpImg = null;
                try {
                    images.put(new WorkItem(tmpString, tmpImg));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            ended = true;
        }
    }

    class Consumer implements Runnable {
        WorkItem tmp = null;
        LinkedList<GlobalFeature> features = new LinkedList<GlobalFeature>();
        int count = 0;
        boolean locallyEnded = false;
        StringBuilder sb = new StringBuilder(1024);

        Consumer() {
            addFeatures(features);
        }

        public void run() {
            while (!locallyEnded) {
                try {
                    // we wait for the stack to be either filled or empty & not being filled any more.
                    // make sure the thread locally knows that the end has come (outer loop)
//                    if (images.peek().getBuffer() == null)
//                        locallyEnded = true;
                    // well the last thing we want is an exception in the very last round.
                    if (!locallyEnded) {
                        tmp = images.take();
                        if (tmp.getBuffer() == null)
                            locallyEnded = true;
                        else {
                            count++;
                            overallCount++;
                        }
                    }

                    if (!locallyEnded) {
                        sb.delete(0, sb.length());
                        ByteArrayInputStream b = new ByteArrayInputStream(tmp.getBuffer());

                        // reads the image. Make sure twelve monkeys lib is in the path to read all jpegs and tiffs.
                        BufferedImage read = ImageIO.read(b);
                        // --------< preprocessing >-------------------------
//                        // converts color space to INT_RGB
                        BufferedImage img = ImageUtils.createWorkingCopy(read);
//                        if (isPreprocessing) {
//                            // despeckle
//                            DespeckleFilter df = new DespeckleFilter();
//                            img = df.filter(img, null);
                        img = ImageUtils.trimWhiteSpace(img); // trims white space
//                        }
                        // --------< / preprocessing >-------------------------

                        if (maxSideLength > 50)
                            img = ImageUtils.scaleImage(img, maxSideLength); // scales image to 512 max sidelength.

                        else if (img.getWidth() < 32 || img.getHeight() < 32) { // image is too small to be worked with, for now I just do an upscale.
                            double scaleFactor = 128d;
                            if (img.getWidth() > img.getHeight()) {
                                scaleFactor = (128d / (double) img.getWidth());
                            } else {
                                scaleFactor = (128d / (double) img.getHeight());
                            }
                            img = ImageUtils.scaleImage(img, ((int) (scaleFactor * img.getWidth())), (int) (scaleFactor * img.getHeight()));
                        }

                        ImageDataProcessor idp = null;
                        try {
                            if (imageDataProcessor != null) {
                                idp = (ImageDataProcessor) imageDataProcessor.newInstance();
                            }
                        } catch (Exception e) {
                            System.err.println("Could not instantiate ImageDataProcessor!");
                            e.printStackTrace();
                        }
                        // --------< creating doc >-------------------------
                        sb.append("<doc>");
                        sb.append("<field name=\"id\">");
                        if (idp == null)
                            sb.append(tmp.getFileName());
                        else
                            sb.append(idp.getIdentifier(tmp.getFileName()));
                        sb.append("</field>");
                        sb.append("<field name=\"title\">");
                        if (idp == null)
                            sb.append(tmp.getFileName());
                        else
                            sb.append(idp.getTitle(tmp.getFileName()));
                        sb.append("</field>");
                        if (idp != null)
                            sb.append(idp.getAdditionalFields(tmp.getFileName()));

                        for (GlobalFeature feature : features) {
                            String featureCode = FeatureRegistry.getCodeForClass(feature.getClass());
                            if (featureCode != null) {
                                feature.extract(img);
                                String histogramField = FeatureRegistry.codeToFeatureField(featureCode);
                                String hashesField = FeatureRegistry.codeToHashField(featureCode);
                                String metricSpacesField = FeatureRegistry.codeToMetricSpacesField(featureCode);

                                sb.append("<field name=\"" + histogramField + "\">");
                                sb.append(Base64.getEncoder().encodeToString(feature.getByteArrayRepresentation()));
                                sb.append("</field>");
                                if (useBitSampling) {
                                    sb.append("<field name=\"" + hashesField + "\">");
                                    sb.append(arrayToString(BitSampling.generateHashes(feature.getFeatureVector())));
                                    sb.append("</field>");
                                }
                                if (useMetricSpaces && MetricSpaces.supportsFeature(feature)) {
                                    sb.append("<field name=\"" + metricSpacesField + "\">");
                                    sb.append(MetricSpaces.generateHashString(feature));
                                    sb.append("</field>");
                                }
                            }
                        }
                        sb.append("</doc>\n");

                        // --------< / creating doc >-------------------------

                        // finally write everything to the stream - in case no exception was thrown..
                        if (!individualFiles) {
                            synchronized (dos) {
                                dos.write(sb.toString().getBytes());
                                // dos.flush();  // flushing takes too long ... better not.
                            }
                        } else {
                            OutputStream mos = new BufferedOutputStream(new FileOutputStream(tmp.getFileName() + "_solr.xml"));
                            mos.write(sb.toString().getBytes());
                            mos.flush();
                            mos.close();
                        }
                    }
//                    if (!individualFiles) {
//                        synchronized (dos) {
//                            dos.write(buffer.toString().getBytes());
//                        }
//                    }
                } catch (Exception e) {
                    System.err.println("Error processing file " + tmp.getFileName());
                    e.printStackTrace();
                }
            }
        }
    }


}