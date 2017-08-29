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
import net.semanticmetadata.lire.solr.FeatureRegistry;
import net.semanticmetadata.lire.solr.HashingMetricSpacesManager;
import net.semanticmetadata.lire.utils.ImageUtils;

import javax.imageio.ImageIO;
import javax.json.*;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.*;
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
import org.w3c.dom.Document;
import org.xml.sax.SAXException;


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
public class ParallelSolrIndexer {
    private final static int maxCacheSize = 250;
    private final static int numberOfThreads = 8;
    private final static int monitoringInterval = 10;
    private final static int maxSideLength = 512;
    private final static boolean useMetricSpaces = false;
    private final static boolean useBitSampling = true;

    private LinkedBlockingQueue<Product> products = new LinkedBlockingQueue<>(maxCacheSize);
    private boolean ended = false;
    private int overallCount = 0;
    private OutputStream dos;
    private Set<Class> featuresSet;

    private File outfile;

    private String datasetPath;

    private List<String> gendersList;
    private List<String> categoriesList;
    private List<String> shopsList;

    private String gender, category;

    private List<String> previousProductsList;
    private List<Product> newProductsList;

    public ParallelSolrIndexer() {
        featuresSet = new HashSet<>();
        featuresSet.add(CEDD.class);
        featuresSet.add(FCTH.class);
        featuresSet.add(JCD.class);
        featuresSet.add(AutoColorCorrelogram.class);

        HashingMetricSpacesManager.init(); // load reference points from disk.

        previousProductsList = new ArrayList<>();
        newProductsList = new ArrayList<>();
    }

    public static void main(String[] args) throws IOException {
        BitSampling.readHashFunctions();
        ParallelSolrIndexer indexer = new ParallelSolrIndexer();

        if (args.length < 4) {
            System.err.println("Wrong number of arguments. 3 needed. Usage example:\n" +
                    "home/path/to/dataset mujer all");
            System.exit(1);
        }

        indexer.setDatasetPath(args[0]);

        indexer.configureGendersList(args[1]);
        if (indexer.gendersList.isEmpty()) {
            System.err.println("Invalid gender argument supplied");
            System.exit(1);
        }
        indexer.configureCategoriesList(args[2]);
        if (indexer.categoriesList.isEmpty()) {
            System.err.println("Invalid category argument supplied");
            System.exit(1);
        }
        indexer.configureShopsList(args[3]);
        if (indexer.shopsList.isEmpty()) {
            System.err.println("Invalid shop argument supplied");
            System.exit(1);
        }

        for (String gender : indexer.gendersList) {
            indexer.gender = gender;
            for (String category : indexer.categoriesList) {
                System.out.println("Empezando con: " + category);
                indexer.category = category;
                String categoryFolderPath = indexer.datasetPath + "/" + gender + "/" + category;

                indexer.initOutfile(categoryFolderPath, category);

                indexer.newProductsList.clear();
                for (String shop: indexer.shopsList) {
                    indexer.configurePreviousProductsList(shop);

                    indexer.configureNewProductsList(shop);
                }

                if (!indexer.previousProductsList.isEmpty()) {
                    indexer.dos.write("<delete>\n".getBytes());
                    indexer.writeIdsToDelete();
                    indexer.dos.write("</delete>\n".getBytes());
                }

                if (!indexer.newProductsList.isEmpty()) {
                    indexer.dos.write("<add>\n".getBytes());
                    indexer.writeDocumentsToAdd();
                    indexer.dos.write("</add>\n".getBytes());
                }

                indexer.closeOutfile();

                if (!indexer.previousProductsList.isEmpty() || !indexer.newProductsList.isEmpty()) {
                    indexer.postIndexToServer();
                }
            }
        }
    }

    private void setDatasetPath(String datasetPath) {
        this.datasetPath = datasetPath;
    }

    private void configureGendersList(String genderArgument) {
        String[] allGenders = { "hombre", "mujer" };

        gendersList = new ArrayList<>();
        if (Arrays.asList(allGenders).contains(genderArgument)) {
            gendersList.add(genderArgument);
        } else {
            if (genderArgument.equals("all")) {
                gendersList.addAll(Arrays.asList(allGenders));
            }
        }
    }

    private void configureCategoriesList(String categoryArgument) {
        String[] allCategories = {
                "abrigos_chaquetas",
                "camisas_blusas",
                "camisetas",
                "faldas",
                "monos",
                "pantalones_cortos",
                "pantalones_largos",
                "punto",
                "sudaderas_jerseis",
                "tops_bodies",
                "vestidos"
        };

        categoriesList = new ArrayList<>();
        if (Arrays.asList(allCategories).contains(categoryArgument)) {
            categoriesList.add(categoryArgument);
        } else {
            if (categoryArgument.equals("all")) {
                categoriesList.addAll(Arrays.asList(allCategories));
            }
        }
    }

    private void configureShopsList(String shopArgument) {
        String[] allShops = {
                "asos",
                "laredoute",
                "mango",
                "pullandbear",
                "zara"
        };

        shopsList = new ArrayList<>();
        if (Arrays.asList(allShops).contains(shopArgument)) {
            shopsList.add(shopArgument);
        } else {
            if (shopArgument.equals("all")) {
                shopsList.addAll(Arrays.asList(allShops));
            }
        }
    }

    private void configurePreviousProductsList(String shop) throws FileNotFoundException {
        String previousProductsFilePath = datasetPath + "/" + gender + "/" + category + "/" + shop + "/products" + "/previous_products.json";
        previousProductsList.addAll(extractListFromJsonFile(previousProductsFilePath));
    }

    private void configureNewProductsList(String shop) throws FileNotFoundException {
        String newProductsFilePath = datasetPath + "/" + gender + "/" + category + "/" + shop + "/products" + "/new_products.json";
        List<String> newProductsIdsList = extractListFromJsonFile(newProductsFilePath);

        for (String productId : newProductsIdsList) {
            Product newProduct = new Product(productId, gender, category, shop);
            newProductsList.add(newProduct);
        }
    }

    private static List<String> extractListFromJsonFile(String filePath) throws FileNotFoundException {
        JsonReader newProductsJsonReader = Json.createReader(new FileReader(filePath));
        JsonArray jsonArray = newProductsJsonReader.readArray();
        newProductsJsonReader.close();
        return jsonArray.getValuesAs(JsonString::getString);
    }

    private void initOutfile(String categoryFolderPath, String category) throws IOException {
        outfile = new File(categoryFolderPath + "/outfile_" + category + ".xml");
        dos = new BufferedOutputStream(new FileOutputStream(outfile), 1024 * 1024 * 8);
    }

    private void closeOutfile() throws IOException {
        dos.close();
    }

    private void writeIdsToDelete() throws IOException {
        for (String productId : previousProductsList) {
            dos.write(("<id>" + productId + "</id>\n").getBytes());
        }
    }

    private void postIndexToServer() {
        try {
            SolrClient client = new HttpSolrClient.Builder("http://54.93.254.52:8983/solr/" + gender + "_" + category).build();
            //SolrClient client = new HttpSolrClient.Builder("http://54.93.254.52:8983/solr/prueba").build();

            DocumentBuilderFactory dbfac = DocumentBuilderFactory.newInstance();
            DocumentBuilder docBuilder = dbfac.newDocumentBuilder();
            Document doc = docBuilder.parse(outfile);
            TransformerFactory tf = TransformerFactory.newInstance();
            Transformer transformer = tf.newTransformer();
            transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
            StringWriter writer = new StringWriter();
            transformer.transform(new DOMSource(doc), new StreamResult(writer));
            String xmlOutput = writer.getBuffer().toString().replaceAll("[\n\r]", "");

            DirectXmlRequest xmlreq = new DirectXmlRequest( "/update", xmlOutput);
            client.request(xmlreq);
            client.commit();
        } catch (SolrServerException | IOException | TransformerException | SAXException | ParserConfigurationException e){
            e.printStackTrace();
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

    private void writeDocumentsToAdd() {
        System.out.println("Extracting features: ");
        for (Class listOfFeature : featuresSet) {
            System.out.println("\t" + listOfFeature.getCanonicalName());
        }

        try {
            ended = false;
            overallCount = 0;
            Thread p = new Thread(new Producer(), "Producer");
            p.start();

            LinkedList<Thread> threads = new LinkedList<>();
            long l = System.currentTimeMillis();
            for (int i = 0; i < numberOfThreads; i++) {
                Thread c = new Thread(new Consumer(), "Consumer-" + i);
                c.start();
                threads.add(c);
            }

            Thread m = new Thread(new Monitoring(), "Monitoring");
            m.start();

            for (Thread thread : threads) {
                thread.join();
            }
            p.join();
            m.join();
            long l1 = System.currentTimeMillis() - l;
            System.out.println("Analyzed " + overallCount + " images in " + l1 / 1000 + " seconds, ~" + (overallCount > 0 ? (l1 / overallCount) : "inf.") + " ms each.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    class Product {
        String productId;
        String gender;
        String category;
        String shop;
        String price;
        byte[] buffer;

        Product() {
            productId = null;
        }

        Product(String productId, String gender, String category, String shop) throws FileNotFoundException {
            this.productId = productId;
            this.gender = gender;
            this.category = category;
            this.shop = shop;

            generateInfoFromJson();
        }

        String getCroppedImagePath() {
            return datasetPath + "/" + this.gender + "/" + this.category + "/CROPPED/" + productId + "_CROPPED.png";
        }

        void generateInfoFromJson() throws FileNotFoundException {
            String jsonFilePath = datasetPath + "/" + this.gender + "/" + this.category + "/" + shop + "/products/" + productId + "/" + productId + ".json";

            JsonReader jsonReader = Json.createReader(new FileReader(jsonFilePath));
            JsonObject jsonObject = jsonReader.readObject();
            jsonReader.close();

            price = jsonObject.getString("price");
        }

        void setBuffer(byte[] buffer) {
            this.buffer = buffer;
        }
    }

    class Monitoring implements Runnable {
        public void run() {
            long ms = System.currentTimeMillis();
            try {
                Thread.sleep(1000 * monitoringInterval);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            while (!ended) {
                try {
                    long time = System.currentTimeMillis() - ms;
                    System.out.println("Analyzed " + overallCount + " images in " + time / 1000 + " seconds, " + ((overallCount > 0) ? (time / overallCount) : "n.a.") + " ms each (" + products.size() + " images currently in queue).");
                    Thread.sleep(1000 * monitoringInterval); // wait xx seconds
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    class Producer implements Runnable {
        public void run() {
            for(Product next: newProductsList) {
                try {
                    File croppedImage = new File(next.getCroppedImagePath());
                    // reading from harddrive to buffer to reduce the load on the HDD and move decoding to the
                    // consumers using java.nio
                    int fileSize = (int) croppedImage.length();
                    byte[] buffer = new byte[fileSize];
                    FileInputStream fis = new FileInputStream(croppedImage);
                    FileChannel channel = fis.getChannel();
                    MappedByteBuffer map = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
                    map.load();
                    map.get(buffer);
                    Product product = new Product(next.productId, next.gender, next.category, next.shop);
                    product.setBuffer(buffer);
                    products.put(product);
                } catch (Exception e) {
                    System.err.println("Could not read image " + next.getCroppedImagePath() + ": " + e.getMessage());
                }
            }
            for (int i = 0; i < numberOfThreads; i++) {
                try {
                    products.put(new Product());
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            ended = true;
        }
    }

    class Consumer implements Runnable {
        Product tmp = null;
        LinkedList<GlobalFeature> features = new LinkedList<>();
        int count = 0;
        boolean locallyEnded = false;
        StringBuilder sb = new StringBuilder(1024);

        Consumer() {
            addFeatures();
        }

        private void addFeatures() {
            for (Class next : featuresSet) {
                try {
                    features.add((GlobalFeature) next.newInstance());
                } catch (InstantiationException | IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
        }

        public void run() {
            while (!locallyEnded) {
                try {
                    if (!locallyEnded) {
                        tmp = products.take();
                        if (tmp.productId == null)
                            locallyEnded = true;
                        else {
                            count++;
                            overallCount++;
                        }
                    }

                    if (!locallyEnded) {
                        sb.delete(0, sb.length());
                        ByteArrayInputStream b = new ByteArrayInputStream(tmp.buffer);

                        // reads the image. Make sure twelve monkeys lib is in the path to read all jpegs and tiffs.
                        BufferedImage read = ImageIO.read(b);
                        // converts color space to INT_RGB
                        BufferedImage img = ImageUtils.createWorkingCopy(read);
                        img = ImageUtils.trimWhiteSpace(img); // trims white space

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

                        // --------< creating doc >-------------------------
                        String imageId =  tmp.productId;
                        sb.append("<doc>");
                        sb.append("<field name=\"id\">");
                        sb.append(imageId);
                        sb.append("</field>");
                        sb.append("<field name=\"shop\">");
                        sb.append(tmp.shop);
                        sb.append("</field>");
                        sb.append("<field name=\"price\">");
                        sb.append(tmp.price);
                        sb.append("</field>");

                        for (GlobalFeature feature : features) {
                            String featureCode = FeatureRegistry.getCodeForClass(feature.getClass());
                            if (featureCode != null) {
                                feature.extract(img);
                                String histogramField = FeatureRegistry.codeToFeatureField(featureCode);
                                String hashesField = FeatureRegistry.codeToHashField(featureCode);
                                String metricSpacesField = FeatureRegistry.codeToMetricSpacesField(featureCode);

                                sb.append("<field name=\"").append(histogramField).append("\">");
                                sb.append(Base64.getEncoder().encodeToString(feature.getByteArrayRepresentation()));
                                sb.append("</field>");
                                if (useBitSampling) {
                                    sb.append("<field name=\"").append(hashesField).append("\">");
                                    sb.append(arrayToString(BitSampling.generateHashes(feature.getFeatureVector())));
                                    sb.append("</field>");
                                }
                                if (useMetricSpaces && MetricSpaces.supportsFeature(feature)) {
                                    sb.append("<field name=\"").append(metricSpacesField).append("\">");
                                    sb.append(MetricSpaces.generateHashString(feature));
                                    sb.append("</field>");
                                }
                            }
                        }
                        sb.append("</doc>\n");
                        // --------< / creating doc >-------------------------

                        // finally write everything to the stream - in case no exception was thrown..
                        synchronized (dos) {
                            dos.write(sb.toString().getBytes());
                        }
                    }
                } catch (Exception e) {
                    System.err.println("Error processing file " + tmp.getCroppedImagePath());
                    e.printStackTrace();
                }
            }
        }
    }


}