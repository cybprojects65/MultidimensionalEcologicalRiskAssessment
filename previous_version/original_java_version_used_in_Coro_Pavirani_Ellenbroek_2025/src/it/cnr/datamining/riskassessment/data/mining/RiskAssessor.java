package it.cnr.datamining.riskassessment.data.mining;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;

import com.rapidminer.RapidMiner;

import it.cnr.datamining.riskassessment.data.clustering.MultiKMeans;
import it.cnr.datamining.riskassessment.data.clustering.XMeans;
import it.cnr.datamining.riskassessment.data.interpretation.ClusterInterpreter;
import it.cnr.datamining.riskassessment.data.interpretation.MedDataInterpreter;
import it.cnr.datamining.riskassessment.data.interpretation.StandardDataInterpreter;
import it.cnr.datamining.riskassessment.data.utils.Operations;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class RiskAssessor {

	boolean multiKmeansActive = false;
	public File outputClusteringModelFile = null;
	public File outputClusteredTable = null;
	public File outputClusterStatsTable = null;
	public File clusteringFile = null;
	public File outputClusterStatsTableQuantized = null;
	public File outputfileofIntepretedVectors = null;
	public File outputTrainingMeansAndVariances = null;
	public ClusterInterpreter interpreter = null;
	public  File featureOutputFolder = null;
	public int minElementsInCluster = 2;
	public int minClusters = 3;
	public int maxClusters = 16;// 50; //9 is the best for multikmeans
	public int maxIterations = 100;
	public String multikmeansFolder = "./multikm_clusters/";
	public String xmeansFolder = "./xmeans_clusters/";
	
	public RiskAssessor(int minElementsInCluster, int minClusters , int maxClusters , int maxIterations, ClusterInterpreter interpreter ) {
		
		this.minElementsInCluster = minElementsInCluster;
		this.minClusters = minClusters;
		this.maxClusters = maxClusters;
		this.maxIterations = maxIterations;
		this.interpreter = interpreter;
		
	}
	
	public static double percentile(double[] vector, double percentile, boolean skipzeros) {

		List<Double> latencies = Arrays.asList(ArrayUtils.toObject(vector));

		Collections.sort(latencies);

		if (skipzeros) {
			List<Double> latenciesnozero = new ArrayList<Double>();
			for (Double lat : latencies) {
				if (lat != 0d)
					latenciesnozero.add(lat);
			}
			latencies = latenciesnozero;
		}
		int index = (int) Math.ceil(percentile / 100.0 * latencies.size());
		return latencies.get(index - 1);
	}
	
	
	public static List<Integer> headers2columnIdx (String header, String[] listOfFeaturesToCluster){
		
		String head_elements[] = header.split(",");
		int headerCounter = 0;
		System.out.println("Selecting headers..");

		List<Integer> validIdxs = new ArrayList<>();
		for (String h : head_elements) {
			for (String feature2cluster : listOfFeaturesToCluster) {
				if (h.equalsIgnoreCase(feature2cluster)) {
					System.out.println("Selecting header: " + h);
					validIdxs.add(headerCounter);
					break;
				}
			}
			headerCounter++;
		}
		
		return validIdxs;
		
	}
	
	public double[][] stringRows2Matrix(List<String> allLines, List<Integer> validIdxs){
		
		int ncells = allLines.size() - 1;
		
		double[][] featureMatrix = new double[ncells][validIdxs.size()];
		
		System.out.println("Building matrix..");
		
		int lineIndex = 1;

		for (int i = 0; i < featureMatrix.length; i++) {

			String row = allLines.get(lineIndex);
			String rowE[] = row.split(",");

			for (int j = 0; j < featureMatrix[0].length; j++) {

				featureMatrix[i][j] = Double.parseDouble(rowE[validIdxs.get(j)]);

			}
			lineIndex++;
		}

		System.out.println("Added " + (lineIndex - 1) + " rows and " + validIdxs.size() + " columns");
		
		return featureMatrix;
		
		
	}
	
	public static boolean RapidMinerInitialised = false;
	public File xMeansClustering(double [][] featureMatrix_std, File featureOutputFolder) throws Exception{
		
		if (!RapidMinerInitialised) {
		System.setProperty("rapidminer.init.operators", "./config/operators.xml");
		RapidMiner.init();
		System.out.println("Rapid Miner initialized");
		
		}
		
		clusteringFile = new File(featureOutputFolder,"clustering.csv");
		if (!featureOutputFolder.exists())
			featureOutputFolder.mkdir();

		CSVLoader loader = new CSVLoader();
		StringBuffer sb = new StringBuffer();
		System.out.println("Clustering "+featureMatrix_std.length+" features");
		for (int i = -1; i < featureMatrix_std.length; i++) {
			for (int j = 0; j < featureMatrix_std[0].length; j++) {
				if (i == -1)
					sb.append("F" + j);
				else
					sb.append(featureMatrix_std[i][j]);
				if (j < (featureMatrix_std[0].length - 1)) {
					sb.append(",");
				} else
					sb.append("\n");
			}
		}
		
		
		InputStream tis = new ByteArrayInputStream(sb.toString().getBytes("UTF-8"));
		loader.setSource(tis);
		//Note: this requires Java 8
		Instances id = loader.getDataSet();
		long ti = System.currentTimeMillis();
		
		System.out.println("XMeans: Clustering ...");
		XMeans xmeans = new XMeans();
		xmeans.setMaxIterations(maxIterations);
		xmeans.setMinNumClusters(minClusters);
		xmeans.setMaxNumClusters(maxClusters);
		xmeans.buildClusterer(id);
		System.out.println("XMEANS: ...ELAPSED CLUSTERING TIME: " + (System.currentTimeMillis() - ti));

		// do clustering
		Instances is = xmeans.getClusterCenters();
		int nClusters = is.numInstances();
		System.out.println("Estimated " + nClusters + " best clusters");
		int[] clusteringAssignments = xmeans.m_ClusterAssignments;
		
		String columnsNames = "id,label,cluster_id,is_an_outlier\n";
		
		StringBuffer bufferRows = new StringBuffer();
		bufferRows.append(columnsNames);
		int nrows = featureMatrix_std.length;
		for (int k = 0; k < nrows; k++) {
			int cindex = clusteringAssignments[k];
			boolean isoutlier = false;
			bufferRows.append((k + 1) + ",F" + (k + 1) + "," + cindex + "," + isoutlier + "\n");
		}
		BufferedWriter bwx = new BufferedWriter(new FileWriter(clusteringFile));
		bwx.write(bufferRows.toString());
		bwx.close();
		return clusteringFile;
	}
	
	public HashMap<Integer, List<Double[]>> assignVectorsToClusterAndSave(double [][] featureMatrix, double [][] fullMatrix, String headerString) throws Exception{
		
		System.out.println("Attach cluster information to vectors");
		int nfeatures = featureMatrix[0].length;
		List<String> clusteredFeatures = Files.readAllLines(clusteringFile.toPath());
		Double[][] clusteredtable = new Double[featureMatrix.length][featureMatrix[0].length];
		HashMap<Integer, List<Double[]>> clustersWithPoints = new HashMap<Integer, List<Double[]>>();
		int addedVectorsToCluster = 0;
		for (int clustidx = 1; clustidx < clusteredFeatures.size(); clustidx++) {

			String clusteringLine = clusteredFeatures.get(clustidx);
			String clusteringLineElements[] = clusteringLine.split(",");
			int featureIdx = Integer.parseInt(clusteringLineElements[0]) - 1;
			int clusterId = Integer.parseInt(clusteringLineElements[2]);

			Double[] row = new Double[nfeatures + 1];
			for (int k = 0; k < nfeatures; k++) {
				row[k] = featureMatrix[featureIdx][k];
			}
			row[nfeatures] = (double) clusterId;

			List<Double[]> pointlist = clustersWithPoints.get(clusterId);

			if (pointlist == null)
				pointlist = new ArrayList<Double[]>();

			pointlist.add(row);
			addedVectorsToCluster++;
			clustersWithPoints.put(clusterId, pointlist);

			Double[] rowcomplete = new Double[fullMatrix[0].length + 1];
			for (int k = 0; k < fullMatrix[0].length; k++) {
				rowcomplete[k] = fullMatrix[featureIdx][k];
			}
			rowcomplete[fullMatrix[0].length] = (double) clusterId;

			clusteredtable[featureIdx] = rowcomplete;
		}
		
		System.out.println("Overall added vectors to clusters "+addedVectorsToCluster);
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputClusteredTable));

		bw.write("longitude,latitude," + headerString + ",clusterid\n");
		for (Double[] row : clusteredtable) {
			int c = 0;
			for (Double r : row) {
				bw.write("" + r);
				if (c < (row.length - 1))
					bw.write(",");
				else
					bw.write("\n");
				c++;
			}
		}

		bw.close();
		System.out.println("Output of clustering is in file " + clusteringFile.getAbsolutePath());
		
		return clustersWithPoints;
	}
	
	public List<double[]> calculateClusterCentroidsAndSave(HashMap<Integer, List<Double[]>> clustersWithPoints, String header) throws Exception{
		
		System.out.println("Calculating clusters' stats");
		double[][] clusterCentroids = null;
		
		for (Integer key : clustersWithPoints.keySet()) {
			
			List<Double[]> clustervectors = clustersWithPoints.get(key);
			int npointspercluster = clustervectors.size();
			int nfeatures = clustervectors.get(0).length-1;
			
			if (clusterCentroids==null)
				clusterCentroids = new double[clustersWithPoints.keySet().size()][nfeatures+1];
			
			
			double[] means = new double[nfeatures + 1];
			means[0] = (double) key;

			for (int i = 1; i < means.length; i++) {
				for (Double[] vector : clustervectors) {
					means[i] = means[i] + vector[i - 1];
				}
				means[i] = means[i] / (double) npointspercluster;
			}

			clusterCentroids [key]= means;
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(outputClusterStatsTable));
		bw.append("clusterid," + header + "\n");

		for (double[] row : clusterCentroids) {
			int c = 0;
			for (double r : row) {
				bw.write("" + r);
				if (c < (row.length - 1))
					bw.write(",");
				else
					bw.write("\n");
				c++;
			}
		}
		bw.close();
		System.out.println("Done.");

		List<double[]> sortedclustercentroids = new ArrayList<>();
		
		for (int i=0;i<clusterCentroids.length;i++) {
			sortedclustercentroids.add(clusterCentroids[i]);
		}
		return sortedclustercentroids;
	}
	
	public List<String[]> quantiseClusters(double [][] featureMatrix, List<double[]> clusterCentroids, String header) throws Exception{
		
		System.out.println("Extracting quantiles to interpret the clusters");

		double[][] featureMatrixT = Operations.traspose(featureMatrix);
		int nfeatures = featureMatrixT.length;
		
		List<String[]> clusterCentroidsQuantised = new ArrayList<String[]>();
		String headers_clust_stat = "clusterid,"
				+ header;
		
		String headers_clust_stat_elements[] = headers_clust_stat.split(",");
		boolean skipzeros = true;
		for (int i = 0; i < featureMatrixT.length; i++) {

			double q2 = percentile(featureMatrixT[i], 25, skipzeros);
			double q4 = percentile(featureMatrixT[i], 75, skipzeros);

			System.out.println("Ranges for " + headers_clust_stat_elements[i + 1] + ": " + q2 + "<x<" + q4);
			int k = 0;
			for (double[] clusterMean : clusterCentroids) {
				String[] clusterCentroidQuantised = null;
				if (i > 0) {
					clusterCentroidQuantised = clusterCentroidsQuantised.get(k);
				}else {
					clusterCentroidQuantised = new String[nfeatures + 1];
					clusterCentroidQuantised[0] = "" + clusterCentroids.get(k)[0];
					clusterCentroidsQuantised.add(clusterCentroidQuantised);
				}
				if (clusterMean[i + 1] <= q2)
					clusterCentroidQuantised[i + 1] = "Low";
				else if (clusterMean[i + 1] >= q4)
					clusterCentroidQuantised[i + 1] = "High";
				else
					clusterCentroidQuantised[i + 1] = "Medium";	
			
				
				k++;
			}
		}

		System.out.println("Clusters have been quantized");
		return clusterCentroidsQuantised;
	}
	
	public File assessOverallClusterRiskAndSave(String header, List<String[]> clusterCentroidsQuantised) throws Exception{
		
		System.out.println("Interpreting clusters as risks");
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputClusterStatsTableQuantized));
		String headers_clust_stat_elements = "clusterid,"+header;
		bw.append(headers_clust_stat_elements+ ",risk\n");

		for (String[] row : clusterCentroidsQuantised) {
			String riskInterpretation = interpreter.interpretCluster(row,headers_clust_stat_elements.split(","));

			int c = 0;
			for (String r : row) {
				bw.write("" + r);
				if (c < (row.length - 1))
					bw.write(",");
				else
					bw.write("," + riskInterpretation + "\n");
				c++;
			}
		}
		bw.close();
		System.out.println("Done.");
		
		return outputClusterStatsTableQuantized;
	}
	
	public static String headerVector2Header(String[] listOfFeaturesToCluster) {
		StringBuffer headerString = new StringBuffer();
		for (String h : listOfFeaturesToCluster)
				headerString.append( h + ",");
		
		String headerFeatures = headerString.toString().substring(0, headerString.toString().length()-1);
		return headerFeatures;
	}
	
	
	public static void saveModelParameters(File output, Operations operations) throws Exception{
		
		FileWriter fw = new FileWriter(output);
		fw.write("Feature_means="+Arrays.toString(operations.means)+"\n");
		fw.write("Feature_variances="+Arrays.toString(operations.variances)+"\n");
		fw.close();
		
		
	}
	
	
	public File clusteriseAndAssessRisk(File inputFile, String[] listOfFeaturesToCluster) throws Exception{

		System.out.println("Clustering and risk classification started");
		List<String> allLines = Files.readAllLines(inputFile.toPath());
		int ncells = allLines.size() - 1;
		System.out.println("Processing vectors associated to "+ncells+" cells");
		String allColumnsHeader = allLines.get(0);
		List<Integer> validIdxs = headers2columnIdx(allColumnsHeader, listOfFeaturesToCluster);
		
		String headerFeatures = headerVector2Header(listOfFeaturesToCluster);
		
		System.out.println("Building matrix..");
		double[][] featureMatrix = stringRows2Matrix(allLines, validIdxs);
		List<Integer> validIdxsPlusLongLat = new ArrayList<>(validIdxs);
		validIdxsPlusLongLat.add(0, 0);
		validIdxsPlusLongLat.add(1, 1);
		double[][] featurePlusLongLatMatrix = stringRows2Matrix(allLines, validIdxsPlusLongLat);
		
		System.out.println("Standardising the dataset..");
		Operations operations = new Operations();
		double[][] featureMatrix_std = operations.standardize(featureMatrix); //featureMatrix;//
		
		if (multiKmeansActive) {
			System.out.println("Applying MultiKmeans..");
			MultiKMeans clusterer = new MultiKMeans();
			featureOutputFolder = new File(multikmeansFolder);
			if (!featureOutputFolder.exists())
				featureOutputFolder.mkdir();
			// retrieve the matrix of the features
			clusteringFile = clusterer.clusterFeatures(featureMatrix_std, featureOutputFolder, minElementsInCluster,
					minClusters, maxClusters);
			
		}else {
			System.out.println("Applying Xmeans..");
			featureOutputFolder = new File(xmeansFolder);
			clusteringFile = xMeansClustering(featureMatrix_std,featureOutputFolder);
		}

		System.out.println("Features have been clustered. The model file is: "+clusteringFile.getAbsolutePath());

		System.out.println("Preparing output files..");
		outputClusteredTable = new File(featureOutputFolder, "clustered_vectors.csv");
		outputClusterStatsTable = new File(featureOutputFolder, "cluster_centroids_numeric.csv");
		outputClusterStatsTableQuantized = new File(featureOutputFolder,"cluster_centroids_classified.csv");
		outputfileofIntepretedVectors = new File(featureOutputFolder,"clustered_vectors_assessed.csv");
		outputTrainingMeansAndVariances = new File(featureOutputFolder,"model_parameters.csv");
		
		saveModelParameters(outputTrainingMeansAndVariances,operations);
		
		System.out.println("Assigning vectors to clusters");
		HashMap<Integer, List<Double[]>> clustersWithPoints = assignVectorsToClusterAndSave(featureMatrix, featurePlusLongLatMatrix, headerFeatures);
		System.out.println("Calculating centroids");
		List<double[]> clusterCentroids = calculateClusterCentroidsAndSave(clustersWithPoints, headerFeatures);
		System.out.println("Quantising centroids");
		List<String[]> clusterCentroidsQuantised =  quantiseClusters(featureMatrix, clusterCentroids, headerFeatures);
		System.out.println("Risk-assing centroids");
		File finalModelOutput = assessOverallClusterRiskAndSave(headerFeatures, clusterCentroidsQuantised);
		System.out.println("Projecting the training set");
		interpreter.interpretTrainingPointGrid(outputClusterStatsTableQuantized, outputClusterStatsTable, outputClusteredTable, outputfileofIntepretedVectors);
		
		System.out.println("\nClustering raw output: "+clusteringFile.getAbsolutePath());
		System.out.println("Clustered input vectors: "+outputClusteredTable.getAbsolutePath());
		System.out.println("Clustering centroids: "+outputClusterStatsTable.getAbsolutePath());
		System.out.println("Clustering centroids quantised and risk-assessed: "+outputClusterStatsTableQuantized.getAbsolutePath());
		System.out.println("Clustering input vectors risk-assessed: "+outputfileofIntepretedVectors.getAbsolutePath());
		
		System.out.println("Done.");

		return finalModelOutput;
	}

	
	public static void main(String[] args) throws Exception {
		
		
		File inputFile = new File(
				"./all_features_space_time_2017_2018_2019_2020_2021_2050.csv");
		String [] features2cluster2019 = {"environment 2019_land_distance","environment 2019_maximum_depth","environment 2019_mean_depth","environment 2019_minimum_depth","environment 2019_net_primary_production","environment 2019_sea-bottom_dissolved_oxygen","environment 2019_sea-bottom_salinity","environment 2019_sea-bottom_temperature","environment 2019_sea-surface_salinity","environment 2019_sea-surface_temperature","fishing activity 2019_reported_fishing","fishing activity 2019_total_fishing","fishing activity 2019_unreported_fishing","species richness 2019","stocks richness 2019"}; 
		 
		int minElementsInCluster = 2;
		int minClusters = 1;
		int maxClusters = 20;// 50; //9 is the best for multikmeans
		int maxIterations = 100;
		System.out.println("Sample Arguments: \nminElementsInCluster (e.g., 2)\nminClusters (e.g., 1)\nmaxClusters (e.g., 20)\nmaxIterations (e.g., 100)\ninputFile (e.g., ./features.csv)\nfeatures2cluster (e.g., \"fishing activity 2019_unreported_fishing,species richness 2019,stocks richness 2019\")\nfeatures2project (e.g., \\\"fishing activity 2019_unreported_fishing,species richness 2019,stocks richness 2019\\\")");
		System.out.println("Example: 2 1 20 100 \"./all_features_space_time_2017_2018_2019_2020_2021_2050.csv\" \"fishing activity 2019_unreported_fishing,species richness 2019,stocks richness 2019\" \"fishing activity 2018_unreported_fishing,species richness 2018,stocks richness 2018\"");
		
		
		if (args!=null && args.length>0) {
			
			minElementsInCluster = Integer.parseInt(args[0]);
			minClusters = Integer.parseInt(args[1]);
			maxClusters = Integer.parseInt(args[2]);
			maxIterations = Integer.parseInt(args[3]);
			
			inputFile = new File(args[4]);
			features2cluster2019 = args[5].split(",");
			
			StandardDataInterpreter interpreter = new StandardDataInterpreter();
			
			RiskAssessor riskassessor = new RiskAssessor(minElementsInCluster, minClusters, maxClusters, maxIterations, interpreter);
			
			File output = riskassessor.clusteriseAndAssessRisk(inputFile, features2cluster2019);
			System.out.println("Final output: "+output.getAbsolutePath());
			
			String [] features2test = args[6].split(",");
			File testOutputOnTrainingSet = new File (riskassessor.featureOutputFolder,"projection_training_set.csv");
			ClusterInterpreter.interpretScenario(riskassessor.outputClusterStatsTableQuantized, riskassessor.outputClusterStatsTable, riskassessor.outputTrainingMeansAndVariances, inputFile, features2test,testOutputOnTrainingSet);
			
			
		}else {
			
		System.out.println("Demo mode..");
		
		MedDataInterpreter interpreter = new MedDataInterpreter();
		
		RiskAssessor riskassessor = new RiskAssessor(minElementsInCluster, minClusters, maxClusters, maxIterations, interpreter);
		
		File output = riskassessor.clusteriseAndAssessRisk(inputFile, features2cluster2019);
		System.out.println("Final output: "+output.getAbsolutePath());
		
		
		File testOutputOnTrainingSet = new File (riskassessor.featureOutputFolder,"projection_training_set_2019.csv");
		ClusterInterpreter.interpretScenario(riskassessor.outputClusterStatsTableQuantized, riskassessor.outputClusterStatsTable, riskassessor.outputTrainingMeansAndVariances, inputFile, features2cluster2019,testOutputOnTrainingSet);
		
		String [] features2cluster2017 = {"environment 2017_land_distance","environment 2017_maximum_depth",
				"environment 2017_mean_depth","environment 2017_minimum_depth",
				"environment 2017_net_primary_production","environment 2017_sea-bottom_dissolved_oxygen",
				"environment 2017_sea-bottom_salinity","environment 2017_sea-bottom_temperature",
				"environment 2017_sea-surface_salinity","environment 2017_sea-surface_temperature",
				"fishing activity 2017_reported_fishing","fishing activity 2017_total_fishing",
				"fishing activity 2017_unreported_fishing","species richness 2017","stocks richness 2017"}; 
		
		testOutputOnTrainingSet = new File (riskassessor.featureOutputFolder,"projection_training_set_2017.csv");
		ClusterInterpreter.interpretScenario(riskassessor.outputClusterStatsTableQuantized, riskassessor.outputClusterStatsTable, riskassessor.outputTrainingMeansAndVariances, inputFile, features2cluster2017,testOutputOnTrainingSet);
		
		String [] features2cluster2018 = {"environment 2018_land_distance","environment 2018_maximum_depth",
				"environment 2018_mean_depth","environment 2018_minimum_depth",
				"environment 2018_net_primary_production","environment 2018_sea-bottom_dissolved_oxygen",
				"environment 2018_sea-bottom_salinity","environment 2018_sea-bottom_temperature",
				"environment 2018_sea-surface_salinity","environment 2018_sea-surface_temperature",
				"fishing activity 2018_reported_fishing","fishing activity 2018_total_fishing",
				"fishing activity 2018_unreported_fishing","species richness 2018","stocks richness 2018"}; 
		
		testOutputOnTrainingSet = new File (riskassessor.featureOutputFolder,"projection_training_set_2018.csv");
		ClusterInterpreter.interpretScenario(riskassessor.outputClusterStatsTableQuantized, riskassessor.outputClusterStatsTable, riskassessor.outputTrainingMeansAndVariances, inputFile, features2cluster2018,testOutputOnTrainingSet);
		
		String [] features2cluster2020 = {"environment 2020_land_distance","environment 2020_maximum_depth",
				"environment 2020_mean_depth","environment 2020_minimum_depth",
				"environment 2020_net_primary_production","environment 2020_sea-bottom_dissolved_oxygen",
				"environment 2020_sea-bottom_salinity","environment 2020_sea-bottom_temperature",
				"environment 2020_sea-surface_salinity","environment 2020_sea-surface_temperature",
				"fishing activity 2020_reported_fishing","fishing activity 2020_total_fishing",
				"fishing activity 2020_unreported_fishing","species richness 2020","stocks richness 2020"}; 
		
		testOutputOnTrainingSet = new File (riskassessor.featureOutputFolder,"projection_training_set_2020.csv");
		ClusterInterpreter.interpretScenario(riskassessor.outputClusterStatsTableQuantized, riskassessor.outputClusterStatsTable, riskassessor.outputTrainingMeansAndVariances, inputFile, features2cluster2020,testOutputOnTrainingSet);
		
		
		String [] features2cluster2021 = {"environment 2021_land_distance","environment 2021_maximum_depth",
				"environment 2021_mean_depth","environment 2021_minimum_depth",
				"environment 2021_net_primary_production","environment 2021_sea-bottom_dissolved_oxygen",
				"environment 2021_sea-bottom_salinity","environment 2021_sea-bottom_temperature",
				"environment 2021_sea-surface_salinity","environment 2021_sea-surface_temperature",
				"fishing activity 2021_reported_fishing","fishing activity 2021_total_fishing",
				"fishing activity 2021_unreported_fishing","species richness 2021","stocks richness 2021"}; 
		
		testOutputOnTrainingSet = new File (riskassessor.featureOutputFolder,"projection_training_set_2021.csv");
		ClusterInterpreter.interpretScenario(riskassessor.outputClusterStatsTableQuantized, riskassessor.outputClusterStatsTable, riskassessor.outputTrainingMeansAndVariances, inputFile, features2cluster2021,testOutputOnTrainingSet);
		
		}
	}

	
}
