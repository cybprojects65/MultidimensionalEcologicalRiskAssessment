package it.cnr.datamining.riskassessment.data.interpretation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import it.cnr.datamining.riskassessment.data.mining.RiskAssessor;

public abstract class ClusterInterpreter {

	public abstract String interpretCluster(String[] cluster_quantised, String[] headers_clust_stat_elements);
	
	public void interpretTrainingPointGrid(File intepretedCentroids, File numericCentroids, File inputTable,File outputfile) throws Exception {

		List<String> centroidsLinesQuantized = Files.readAllLines(intepretedCentroids.toPath());
		List<String> centroidsLinesNumeric = Files.readAllLines(numericCentroids.toPath());
		System.out.println("Getting centroids..");
		HashMap<String, String> clusterClassifications = new HashMap<>();
		HashMap<String, double[]> clusterCentroids = new HashMap<>();
		int linenumber = 0;
		for (String centroidLine : centroidsLinesQuantized) {
			if (linenumber > 0) {
				String elements[] = centroidLine.split(",");
				String clusterid = elements[0];
				String clusterClassification = elements[elements.length - 1];
				clusterClassifications.put(clusterid, clusterClassification);

				String elementsNumeric[] = centroidsLinesNumeric.get(linenumber).split(",");

				double[] clusterCentroidvector = new double[elementsNumeric.length - 1];
				for (int i = 1; i < (elementsNumeric.length); i++) {
					clusterCentroidvector[i - 1] = Double.parseDouble(elementsNumeric[i]);
				}
				clusterCentroids.put(clusterid, clusterCentroidvector);
			}
			linenumber++;
		}
		System.out.println("Classifying table elements..");
		List<String> tableLines = Files.readAllLines(inputTable.toPath());
		List<String> newTableLines = new ArrayList<>();
		linenumber = 0;
		for (String tableLine : tableLines) {
			String classification = "";
			if (linenumber == 0) {
				classification = "risk";

			} else {

				String elementsTableLine[] = tableLine.split(",");
				String clusterid = elementsTableLine[elementsTableLine.length - 1];
				classification = clusterClassifications.get(clusterid);
			}
			String newTableLine = tableLine + "," + classification;

			newTableLines.add(newTableLine);

			linenumber++;
		}

		System.out.println("Writing output to "+outputfile.getAbsolutePath()+"..");
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputfile));

		for (String newTableLine : newTableLines) {

			bw.append(newTableLine + "\n");

		}

		bw.close();
		
		System.out.println("Done.");
	}
	
	public static double[] toVector(String line) throws Exception{
		String vectorS = line.substring(line.indexOf("=")+1);
		vectorS=vectorS.replace("[", "").replace("]", "");
		
		String vectorE [] = vectorS.split(",");
		double [] v = new double[vectorE.length];
		int i = 0;
		for (String vE:vectorE) {
			v [i] = Double.parseDouble(vE);
			i++;
		}
		return v;
	}
	
	public static void interpretScenario(File intepretedCentroids, File numericCentroids, File modelParameters, File inputTable,String [] features2cluster, File outputfile) throws Exception {

		List<String> centroidsLinesQuantized = Files.readAllLines(intepretedCentroids.toPath());
		List<String> centroidsLinesNumeric = Files.readAllLines(numericCentroids.toPath());
		List<String> modelParametersLines = Files.readAllLines(modelParameters.toPath());
		double [] vectorMeans = toVector(modelParametersLines.get(0));
		double [] vectorVariances = toVector(modelParametersLines.get(1));
		
		System.out.println("Getting centroids..");
		HashMap<String, String> clusterClassifications = new HashMap<>();
		HashMap<String, double[]> clusterCentroids = new HashMap<>();
		int linenumber = 0;
		for (String centroidLine : centroidsLinesQuantized) {
			if (linenumber > 0) {
				String elements[] = centroidLine.split(",");
				String clusterid = elements[0];
				String clusterClassification = elements[elements.length - 1];
				clusterClassifications.put(clusterid, clusterClassification);

				String elementsNumeric[] = centroidsLinesNumeric.get(linenumber).split(",");

				double[] clusterCentroidvector = new double[elementsNumeric.length - 1];
				for (int i = 1; i < (elementsNumeric.length); i++) {
					clusterCentroidvector[i - 1] = Double.parseDouble(elementsNumeric[i]);
				}
				clusterCentroids.put(clusterid, clusterCentroidvector);
			}
			linenumber++;
		}
		
		String headerFeatures = RiskAssessor.headerVector2Header(features2cluster);
		
		System.out.println("Classifying table elements..");
		List<String> tableLines = Files.readAllLines(inputTable.toPath());
		List<String> newTableLines = new ArrayList<>();
		
		linenumber = 0;
		List<Integer> featureIndices = null;
		
		for (String tableLine : tableLines) {
			String classification = "";
			String newTableLine = "";
			if (linenumber == 0) {
				classification = "risk";
				featureIndices = RiskAssessor.headers2columnIdx(tableLine, features2cluster);
				newTableLine = "longitude,latitude,"+headerFeatures+",clusterid,risk";
			} else {
				String elementsTableLine[] = tableLine.split(",");
				String longitude = elementsTableLine[0];
				String latitude = elementsTableLine[1];
				
				double [] vector = new double[featureIndices.size()];
				int vectorIdx = 0;
				StringBuffer sb = new StringBuffer();
				for (Integer fidx : featureIndices) {
					vector[vectorIdx] = Double.parseDouble(elementsTableLine[fidx]);
					sb.append(""+vector[vectorIdx]);
					if (vectorIdx<(featureIndices.size()-1))
						sb.append(",");	
					vectorIdx++;
				}
				
				String clusterid = findClosestCluster(vector,vectorVariances,clusterCentroids);
				classification = clusterClassifications.get(clusterid);
				newTableLine = longitude+","+latitude+","+sb.toString() +","+clusterid+ "," + classification;
				//System.exit(0);
			}
			newTableLines.add(newTableLine);

			linenumber++;
		}

		System.out.println("Writing output to "+outputfile.getAbsolutePath());
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputfile));

		for (String newTableLine : newTableLines) {

			bw.append(newTableLine + "\n");

		}

		bw.close();
		
		System.out.println("Done.");
	}

	private static String findClosestCluster(double[] vector, double [] vectorVariances, HashMap<String, double[]> clusterCentroids) {
		
		String clusterid = "";
		double mindistance = Double.MAX_VALUE;
		
		//System.out.println(Arrays.toString(vector)+"->");
		for(String key:clusterCentroids.keySet()) {
			
			double [] centroid = clusterCentroids.get(key);
			
			double sumsqr = 0d;
			
			for (int j=0;j<centroid.length;j++) {
				
				sumsqr = sumsqr + ( (centroid[j]-vector[j])*(centroid[j]-vector[j]) /( vectorVariances[j]*vectorVariances[j] ) );
			
			}
			
			double distance = Math.sqrt(sumsqr);
			//System.out.println("d(v,c_"+key+"): "+distance);
			if (distance<mindistance) {
				mindistance = distance;
				clusterid = key;
			}
				
		}
		
		
		return clusterid;
	}
	
}
