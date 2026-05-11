package it.cnr.datamining.riskassessment.data.utils;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Operations {

	public static double scalarProduct(double[] a, double[] b) {
		
		double sum = 0;

		for (int i = 0; i < a.length; i++) {
			if (i < b.length)
				sum = sum + a[i] * b[i];
		}

		return sum;
	}

	public static double sumVector(double[] a) {

		double sum = 0;

		for (int i = 0; i < a.length; i++) {
			sum = sum + a[i];
		}

		return sum;
	}

	public static double sumNormalVector(double[] a, double norm) {

		double sum = 0;

		for (int i = 0; i < a.length; i++) {
			sum = sum + (a[i]/norm);
		}

		
		return sum;
	}
	
	public static double[] vectorialDifference(double[] a, double[] b) {

		double[] diff = new double[a.length];

		for (int i = 0; i < a.length; i++) {
			if (i < b.length)
				diff[i] = a[i] - b[i];
			else
				diff[i] = a[i];
		}

		return diff;
	}

	public static double[] vectorialAbsoluteDifference(double[] a, double[] b) {

		double[] diff = new double[a.length];

		for (int i = 0; i < a.length; i++) {
			if (i < b.length)
				diff[i] = Math.abs(a[i] - b[i]);
			else
				diff[i] = Math.abs(a[i]);
		}

		return diff;
	}

	public static double getMax(double[] points) {
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			if (max < points[i])
				max = points[i];
		}
		return max;
	}

	public static int getMax(int[] points) {
		int max = -Integer.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			if (max < points[i])
				max = points[i];
		}
		return max;
	}

	public static int getMin(int[] points) {
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			if (min > points[i])
				min = points[i];
		}
		return min;
	}

	public static double getMin(double[] points) {
		double min = Double.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			if (min > points[i])
				min = points[i];
		}
		return min;
	}

	// calculates the frequency distribution for a set of points respect to a set of intervals
	public static double[] calcFrequencies(double[] interval, double[] points) {
		int intervs = interval.length;
		int npoints = points.length;
		double[] frequencies = new double[intervs];
		for (int i = 0; i < intervs; i++) {

			for (int j = 0; j < npoints; j++) {

				if (((i == 0) && (points[j] < interval[i])) || ((i == intervs - 1) && (points[j] >= interval[i - 1]) && (points[j] <= interval[i])) || ((i > 0) && (points[j] >= interval[i - 1]) && (points[j] < interval[i]))) {
					// System.out.println("(" + (i == 0 ? "" : interval[i - 1]) + "," + interval[i] + ")" + " - " + points[j]);
					frequencies[i] = frequencies[i] + 1;
				}
			}
		}

		return frequencies;
	}

	public static double[] normalizeFrequencies(double[] frequencies, int numberOfPoints) {
		int intervs = frequencies.length;
		for (int i = 0; i < intervs; i++) {
			if (numberOfPoints>0)
				frequencies[i] = frequencies[i] / (double) numberOfPoints;
			else
				frequencies[i] = 0;
		}

		return frequencies;

	}

	public static double covariance(double[] x, double[] y) {
        double c = 0;
        double meanX = Operations.mean(x);
        double meanY = Operations.mean(y);
        int xl = Math.min(x.length, y.length);
        
        for (int t = 0; t < xl; t++) {
            c += (x[t] - meanX) * (y[t] - meanY);
        }
        return c / (double) (x.length - 1); // -1 for sample covariance
    }
	
	
	public static double[] normalize(double[] vector) {
		double max = getMax(vector);
		
		int intervs = vector.length;
		for (int i = 0; i < intervs; i++) {
			vector[i] = vector[i] / (double) max;
			
		}

		return vector;

	}
	public static double[] [] permuteColumns(double[][] matrix, int i,int j) {
		
		
		
		double [][] permMatrix = new double[matrix.length][matrix[0].length];
		for (int id = 0; id < permMatrix.length; id++)
		     permMatrix[id] = Arrays.copyOf(matrix[id], matrix[id].length);
		
		for (int k=0;k<matrix.length;k++) {
			permMatrix [k] [i]= matrix[k][j];
		}
		
		for (int k=0;k<matrix.length;k++) {
			permMatrix [k] [j]= matrix[k][i];
		}
		
		return permMatrix;

	}
	
	public static double[] normals(double[][] mat) {
		int rows = mat.length;
		int cols = mat[0].length;
		double matnorm [][] = new double[rows][cols]; 
		double maxs [] = new double [cols];
		for (int i = 0;i<maxs.length;i++) {
			maxs[i] = -Double.MAX_VALUE;
		}
		
		for (int i = 0; i < rows; i++) {
		
			for (int j = 0; j < cols; j++) {
				if (mat[i][j]>maxs[j])
					maxs[j] = mat[i][j];
			}
		}
		
		return maxs;
	}
	
	public static double[] normals(double[][] mat1,double[][] mat2) {
		int rows = mat1.length;
		int cols = mat1[0].length;
		
		double maxs [] = new double [cols];
		for (int i = 0;i<maxs.length;i++) {
			maxs[i] = -Double.MAX_VALUE;
		}
		
		for (int i = 0; i < rows; i++) {
		
			for (int j = 0; j < cols; j++) {
				if (mat1[i][j]>maxs[j])
					maxs[j] = mat1[i][j];
			}
		}
		
		for (int i = 0; i < rows; i++) {
			
			for (int j = 0; j < cols; j++) {
				if (mat2[i][j]>maxs[j])
					maxs[j] = mat2[i][j];
			}
		}
		
		return maxs;
	}
	
	public static double[] variances(double[][] mat1,double[][] mat2) {
		int rows = mat1.length;
		int cols = mat1[0].length;
		
		double maxs [] = new double [cols];
		for (int i = 0;i<maxs.length;i++) {
			maxs[i] = -Double.MAX_VALUE;
		}
		
		for (int i = 0; i < rows; i++) {
		
			for (int j = 0; j < cols; j++) {
				if (mat1[i][j]>maxs[j])
					maxs[j] = mat1[i][j];
			}
		}
		
		for (int i = 0; i < rows; i++) {
			
			for (int j = 0; j < cols; j++) {
				if (mat2[i][j]>maxs[j])
					maxs[j] = mat2[i][j];
			}
		}
		
		return maxs;
	}
	
	public static double[][] normalizeMatrix(double[][] mat, double [] maxs) {
		int rows = mat.length;
		int cols = mat[0].length;
		double matnorm [][] = new double[rows][cols]; 
		
		for (int i = 0; i < rows; i++) {
			
			for (int j = 0; j < cols; j++) {
				matnorm[i][j]=mat[i][j]/maxs [j];
			}
			
		}
		return matnorm;

	}
	
	
	public static double[][] normalizeMatrix(double[][] mat) {
		int rows = mat.length;
		int cols = mat[0].length;
		double matnorm [][] = new double[rows][cols]; 
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < rows; i++) {
		
			for (int j = 0; j < cols; j++) {
				if (mat[i][j]>max)
					max = mat[i][j];
			}
			
		}
		
		for (int i = 0; i < rows; i++) {
			
			for (int j = 0; j < cols; j++) {
				matnorm[i][j]=mat[i][j]/max;
			}
			
		}
		return matnorm;

	}
	
	
	// checks if an interval contains at least one element from a sequence of points
	public static boolean intervalContainsPoints(double min, double max, double[] points) {
		// System.out.println(min+"-"+max);
		boolean contains = false;
		for (int i = 0; i < points.length; i++) {
			if ((points[i] >= min) && (points[i] < max)) {
				// System.out.println("---->"+points[i]);
				contains = true;
				break;
			}
		}
		return contains;
	}

	// finds the best subdivision for a sequence of numbers
	public static double[] uniformDivide(double max, double min, double[] points, int maxNofBins) {
		int maxintervals = maxNofBins;
		int n = maxintervals;

		boolean subdivisionOK = false;
		double gap = (max - min) / n;

		// search for the best subdivision: find the best n
		while (!subdivisionOK) {
			// System.out.println("*************************");
			boolean notcontains = false;
			// take the gap interval to test
			for (int i = 0; i < n; i++) {
				double rightmost = 0;
				// for the last border take a bit more than max
				if (i == n - 1)
					rightmost = max + 0.01;
				else
					rightmost = min + gap * (i + 1);
				// if the interval doesn't contain any point discard the subdivision
				if (!intervalContainsPoints(min + gap * i, rightmost, points)) {
					notcontains = true;
					break;
				}
			}

			// if there are empty intervals and there is space for another subdivision proceed
			if (notcontains && n > 0) {
				n--;
				gap = (max - min) / n;
			}
			// otherwise take the default subdivision
			else if (n == 0) {
				n = maxintervals;
				subdivisionOK = true;
			}
			// if all the intervals are non empty then exit
			else
				subdivisionOK = true;
		}

		// once the best n is found build the intervals
		double[] intervals = new double[n];
		for (int i = 0; i < n; i++) {
			if (i < n - 1)
				intervals[i] = min + gap * (i + 1);
			else
				intervals[i] = Double.POSITIVE_INFINITY;
		}

		return intervals;
	}

	public double[][] standardize(double[][] matrix) {
		return standardize(matrix, null, null);
	}

	public double[] means;
	public double[] variances;

	// gets all the columns from a matrix
		public static double[][] traspose(double[][] matrix) {
			int m = matrix.length;
			if (m > 0) {
				int n = matrix[0].length;

				double columns[][] = new double[n][m];

				for (int i = 0; i < n; i++) {
					for (int j = 0; j < m; j++)
						columns[i][j] = matrix[j][i];
				}

				return columns;
			} else
				return null;
		}
		
	// standardizes a matrix: each row represents a vector: outputs columns means and variances
	public double[][] standardize(double[][] matrix, double[] meansVec, double[] variancesVec) {

		if (matrix.length > 0) {
			int ncols = matrix[0].length;
			int mrows = matrix.length;

			if ((means == null) && (variances == null)) {
				means = new double[ncols];
				variances = new double[ncols];
			}

			double[][] matrixT = traspose(matrix);

			for (int i = 0; i < ncols; i++) {
				double[] icolumn = matrixT[i];

				double mean = 0;

				if (meansVec == null) {
					mean = mean(icolumn);
					means[i] = mean;
				} else
					mean = meansVec[i];

				double variance = 0;
				if (variancesVec == null) {
					variance = variance(icolumn);//com.rapidminer.tools.math.MathFunctions.variance(icolumn, Double.NEGATIVE_INFINITY);
					variances[i] = variance;
				} else
					variance = variancesVec[i];

				for (int j = 0; j < mrows; j++) {
					// standardization
					double numerator = (icolumn[j] - mean);
					if ((numerator == 0) && (variance == 0))
						icolumn[j] = 0;
					else if (variance == 0)
						icolumn[j] = Double.MAX_VALUE;
					else
						icolumn[j] = numerator / variance;
				}
			}

			matrix = traspose(matrixT);

		}
		return matrix;
	}

	public static double[][] symmetrizeByMean(double[][] matrix) {
		double [][] newMatrix = new double[matrix.length][matrix[0].length];
		
		for (int i=0;i<newMatrix.length;i++) {
		
			for (int j=0;j<newMatrix[0].length;j++) {
				
				newMatrix [i][j]= (matrix[i][j]+matrix[j][i])/2;
			
			}
			
		}
		
		return newMatrix;
		
	}
	
	public static double[][] symmetrizeByMax(double[][] matrix) {
		double [][] newMatrix = new double[matrix.length][matrix[0].length];
		
		for (int i=0;i<newMatrix.length;i++) {
		
			for (int j=0;j<newMatrix[0].length;j++) {
				
				newMatrix [i][j]= Math.max(matrix[i][j],matrix[j][i]);
			
			}
			
		}
		
		return newMatrix;
		
	}
	
	public static double variance (double[] data) {
		
		double mean = 0.0;
		for (int i = 0; i < data.length; i++) {
		        mean += data[i];
		}
		mean /= (double)data.length;

		// The variance
		double variance = 0;
		for (int i = 0; i < data.length; i++) {
		    variance += Math.pow(data[i] - mean, 2);
		}
		variance /= (double)(data.length-1d);

		// Standard Deviation
		double std = Math.sqrt(variance);
		
		return std;
		
	}
	// calculates the number of elements to take from a set with inverse weight respect to the number of elements
	public static int calcNumOfRepresentativeElements(int numberOfElements, int minimumNumberToTake) {
		return (int) Math.max(minimumNumberToTake, numberOfElements / Math.log10(numberOfElements));
	}

	public static double[] linearInterpolation(double el1, double el2, int intervals) {

		double step = (el2 - el1) / (double) intervals;

		double[] intervalsd = new double[intervals];
		intervalsd[0] = el1;
		for (int i = 1; i < intervals - 1; i++) {
			intervalsd[i] = el1 + step * i;
		}
		intervalsd[intervals - 1] = el2;

		return intervalsd;
	}

	private static double parabol(double a, double b, double c, double x, double shift) {
		return a * (x - shift) * (x - shift) + b * (x - shift) + c;
	}

	public static double[] inverseParabol(double a, double b, double c, double y) {

		double[] ret = { (-1d * b + Math.sqrt(b * b + 4 * a * (Math.abs(y) - c))) / (2 * a), (-1d * b - Math.sqrt(b * b + 4 * a * (Math.abs(y) - c))) / (2 * a) };
		return ret;
	}

	public static double logaritmicTransformation(double y) {
		y = Math.abs(y);
		if (y == 0)
			return -Double.MAX_VALUE;
		else
			return Math.log10(y);
	}

	// the parabol is centered on the start Point
	public static double[] parabolicInterpolation(double startP, double endP, int intervals) {

		double start = startP;
		double end = endP;
		double shift = start;

		double a = 1000d;
		double b = 0d;
		double c = 0d;
		double parabolStart = parabol(a, b, c, start, shift);
		if (start < 0)
			parabolStart = -1 * parabolStart;

		double parabolEnd = parabol(a, b, c, end, start);
		if (end < 0)
			parabolEnd = -1 * parabolEnd;

		double step = 0;
		if (intervals > 0) {
			double difference = Math.abs(parabolEnd - parabolStart);
			step = (difference / (double) intervals);
		}

		double[] linearpoints = new double[intervals];

		linearpoints[0] = startP;
		// System.out.println("Y0: "+parabolStart);
		for (int i = 1; i < intervals - 1; i++) {
			double ypoint = 0;
			if (end > start)
				ypoint = parabolStart + (i * step);
			else
				ypoint = parabolStart - (i * step);
			// System.out.println("Y: "+ypoint);
			double res[] = inverseParabol(a, b, c, Math.abs(ypoint));
			// System.out.println("X: "+linearpoints[i]);
			if (ypoint < 0)
				linearpoints[i] = res[1] + shift;
			else
				linearpoints[i] = res[0] + shift;
		}

		linearpoints[intervals - 1] = endP;
		return linearpoints;
	}

	public static void main1(String[] args) {
		// double [] points = {1,1.2,1.3,2,5};
		double[] points = new double[20];
		for (int i = 0; i < 20; i++)
			points[i] = 10 * Math.random();

		double max = getMax(points);
		double min = getMin(points);
		System.out.println("<" + min + "," + max + ">");

		double[] interval = uniformDivide(max, min, points,10);

		double[] frequencies = calcFrequencies(interval, points);
		for (int i = 0; i < interval.length; i++) {
			System.out.print(interval[i] + " ");
			System.out.println("->" + frequencies[i] + " ");
		}
	}

	public static void main2(String[] args) {
		/*
		 * System.out.println("numbers to take: " + calcNumOfRepresentativeElements(100, 100)); double[] interp = linearInterpolation(27.27, 28.28, 3); double[] parabinterp = parabolicInterpolation(1, 10, 9); System.out.println("");
		 */
		int[] ii = takeChunks(11549, 11549/100);
		System.out.println("OK");
	}

	//distributes uniformly elements in parts
	public static int[] takeChunks(int numberOfElements, int partitionFactor) {
		int[] partitions = new int[1];
		if (partitionFactor <= 0) {
			return partitions;
		} else if (partitionFactor == 1) {
			partitions[0] = numberOfElements;
			return partitions;
		}

		int chunksize = numberOfElements / (partitionFactor);
		int rest = numberOfElements % (partitionFactor);
		if (chunksize == 0) {
			partitions = new int[numberOfElements];
			for (int i = 0; i < numberOfElements; i++) {
				partitions[i] = 1;
			}
		} else {
			partitions = new int[partitionFactor];
			for (int i = 0; i < partitionFactor; i++) {
				partitions[i] = chunksize;
			}

			for (int i = 0; i < rest; i++) {
				partitions[i]++;
			}

		}

		return partitions;
	}

	public static int chunkize(int numberOfElements, int partitionFactor) {
		int chunksize = numberOfElements / partitionFactor;
		int rest = numberOfElements % partitionFactor;
		if (chunksize == 0)
			chunksize = 1;
		else if (rest != 0)
			chunksize++;
		/*
		 * int numOfChunks = numberOfElements / chunksize; if ((numberOfElements % chunksize) != 0) numOfChunks += 1;
		 */

		return chunksize;
	}

	
	public static double[] uniformSampling(double min, double max, int maxElementsToTake){
		double step = (max-min)/(double)(maxElementsToTake-1);
		double [] samples = new double [maxElementsToTake];
		
		for (int i=0;i<samples.length;i++){
			double value = min+i*step;
			if (value>max)
				value=max;
			samples [i] = value;
		}
		
		return samples;
	}
	
	public static int[] uniformIntegerSampling(double min, double max, int maxElementsToTake){
		double step = (max-min)/(double)(maxElementsToTake-1);
		int [] samples = new int [maxElementsToTake];
		
		for (int i=0;i<samples.length;i++){
			double value = min+i*step;
			if (value>max)
				value=max;
			samples [i] = (int)value;
		}
		
		return samples;
	}
	
	public static void main(String[] args) {
		double [] samples = uniformSampling(0, 9, 10);
		System.out.println("OK");
	}
	
	
	//rounds to the xth decimal position
	public static double roundDecimal(double number,int decimalposition){
		
		double n = (double)Math.round(number * Math.pow(10.00,decimalposition))/Math.pow(10.00,decimalposition);
		return n;
	}
	
	// increments a percentage o mean calculation when a lot of elements are present
	public static float incrementPerc(float perc, float quantity, int N) {

		if (N == 0)
			return quantity;

		float out = 0;
		int N_plus_1 = N + 1;
		out = (float) ((perc + ((double) quantity / (double) N)) * ((double) N / (double) N_plus_1));
		return out;

	}

	public static double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix) {
	    double[][] result = new double[firstMatrix.length][secondMatrix[0].length];

	    for (int row = 0; row < result.length; row++) {
	        for (int col = 0; col < result[row].length; col++) {
	            result[row][col] = multiplyMatricesCell(firstMatrix, secondMatrix, row, col);
	        }
	    }

	    return result;
	}
	
	public static double multiplyMatricesCell(double[][] firstMatrix, double[][] secondMatrix, int row, int col) {
	    double cell = 0;
	    for (int i = 0; i < secondMatrix.length; i++) {
	        cell += firstMatrix[row][i] * secondMatrix[i][col];
	    }
	    return cell;
	}
	
	
	public static ArrayList<Integer> generateRandoms(int numberOfRandoms, int min, int max) {

		ArrayList<Integer> randomsSet = new ArrayList<Integer>();
		// if number of randoms is equal to -1 generate all numbers
		if (numberOfRandoms == -1) {
			for (int i = min; i < max; i++) {
				randomsSet.add(i);
			}
		} else {
			int numofrandstogenerate = 0;
			if (numberOfRandoms <= max) {
				numofrandstogenerate = numberOfRandoms;
			} else {
				numofrandstogenerate = max;
			}

			if (numofrandstogenerate == 0) {
				randomsSet.add(0);
			} else {
				for (int i = 0; i < numofrandstogenerate; i++) {

					int RNum = -1;
					RNum = (int) ((max) * Math.random()) + min;

					// generate random number
					while (randomsSet.contains(RNum)) {
						RNum = (int) ((max) * Math.random()) + min;
						// AnalysisLogger.getLogger().debug("generated " + RNum);
					}

					// AnalysisLogger.getLogger().debug("generated " + RNum);

					if (RNum >= 0)
						randomsSet.add(RNum);
				}

			}
		}

		return randomsSet;
	}

	public static int[] generateSequence(int elements) {
		int[] sequence = new int[elements];
		for (int i = 0; i < elements; i++) {
			sequence[i] = i;
		}
		return sequence;
	}

	public static BigInteger chunk2Index(int chunkIndex, int chunkSize) {

		return BigInteger.valueOf(chunkIndex).multiply(BigInteger.valueOf(chunkSize));

	}

	// calculates mean
	public static double mean(double[] p) {
		double sum = 0; // sum of all the elements
		for (int i = 0; i < p.length; i++) {
			sum += p[i];
		}
		return sum / p.length;
	}// end method mean

	//calculates normalized derivative
	public static double[] derivative(double[] a) {
		double[] d = new double[a.length];
		double max = 1;
		if (a.length > 0) {
			for (int i = 0; i < a.length; i++) {
				double current = a[i];
				double previous = current;
				if (i > 0) {
					previous = a[i - 1];
				}
				d[i] = current - previous;
				if (Math.abs(d[i])>max)
					max = Math.abs(d[i]); 
				// System.out.println("point "+a[i]+" derivative "+d[i]);
			}
			
			//normalize
			for (int i = 0; i < a.length; i++) {
				d[i] = d[i]/max;
			}
		}

		return d;
	}

	// returns a list of spikes indexes
	public static boolean[] findMaxima(double[] derivative,double threshold) {
			boolean[] d = new boolean[derivative.length];

			if (d.length > 0) {
				d[0] = false;
				for (int i = 1; i < derivative.length - 1; i++) {
					if ((derivative[i] / derivative[i + 1] < 0) && derivative[i]>0){
//						double ratio = Math.abs((double) derivative[i]/ (double) derivative[i+1]);
//						System.out.println("RATIO "+i+" "+Math.abs(derivative[i]));
//						if ((threshold>0)&&(ratio<threshold))
						if ((threshold>0)&&(Math.abs(derivative[i])>threshold))
							d[i] = true;
					}
					else
						d[i] = false;
				}
				double max = Operations.getMax(derivative);
				if (max==derivative[derivative.length - 1])
					d[derivative.length - 1] = true;
				else
					d[derivative.length - 1] = false;
			}

			return d;
		}
		
	// returns a list of spikes indexes
	public static boolean[] findSpikes(double[] derivative,double threshold) {
		boolean[] d = new boolean[derivative.length];

		if (d.length > 0) {
			d[0] = false;
			for (int i = 1; i < derivative.length - 1; i++) {
				if (derivative[i] / derivative[i + 1] < 0){
//					double ratio = Math.abs((double) derivative[i]/ (double) derivative[i+1]);
//					System.out.println("RATIO "+i+" "+Math.abs(derivative[i]));
//					if ((threshold>0)&&(ratio<threshold))
					if ((threshold>0)&&(Math.abs(derivative[i])>threshold))
						d[i] = true;
				}
				else
					d[i] = false;
			}
			d[derivative.length - 1] = false;
		}

		return d;
	}

	// returns a list of spikes indexes
	public static boolean[] findSpikes(double[] derivative) {
		return findSpikes(derivative,-1);
	}
	

	// searches for an index into an array
	public static boolean isIn(List<Integer> indexarray, int index) {
		
		int size = indexarray.size();
		
		for (int i = 0; i < size; i++) {
			if (index == indexarray.get(i).intValue())
				return true;
		}
		
		return false;
	}
	
	
	// finds the indexes of zero points
	public static List<Integer> findZeros(double[] points) {
		
		int size = points.length;
		ArrayList<Integer> zeros = new ArrayList<Integer>();
		
		for (int i = 0; i < size; i++) {
			if (points[i]==0){
				int start = i;
				int end = i;
				
				for (int j=i+1;j<size;j++)
				{
					if (points[j]!=0){
						end = j-1;
						break;
					}
				}
				int center = start+((end-start)/2); 
				zeros.add(center);
				i = end;
			}
		}
		
		return zeros;
		
	}
	
	
	public static double[] logSubdivision(double start,double end,int numberOfParts){
		
		
		if (end<=start)
			return null;
		
		if (start == 0){
			start = 0.01;
		}
		double logStart = Math.log(start);
		double logEnd = Math.log(end);
		double step =0 ;
		if (numberOfParts >0){
			
			double difference = logEnd-logStart;
			step = (difference/(double)numberOfParts);
			
		}
//		double [] points = new double[numberOfParts+1];
		double[] linearpoints = new double[numberOfParts+1];
		
		for (int i=0;i<numberOfParts+1;i++){
			
//			points[i] = logStart+(i*step);
			
			linearpoints[i]= Math.exp(logStart+(i*step));
			if (linearpoints[i]<0.011)
				linearpoints[i] = 0;
		}
		
		return linearpoints;
	}
	
	
	public static double cohensKappaForDichotomy(long NumOf_A1_B1, long NumOf_A1_B0, long NumOf_A0_B1, long NumOf_A0_B0){
		long  T = NumOf_A1_B1+NumOf_A1_B0+NumOf_A0_B1+NumOf_A0_B0;
		
		double Pra = (double)(NumOf_A1_B1+NumOf_A0_B0)/(double) T ;
		double Pre1 = (double) (NumOf_A1_B1+NumOf_A1_B0) * (double) (NumOf_A1_B1+NumOf_A0_B1)/(double) (T*T);
		double Pre2 = (double) (NumOf_A0_B0+NumOf_A0_B1) * (double) (NumOf_A0_B0+NumOf_A1_B0)/(double) (T*T);
		double Pre = Pre1+Pre2;
		double Kappa = (Pra-Pre)/(1d-Pre);
		return roundDecimal(Kappa,3);
	}
	
	public static String kappaClassificationLandisKoch(double kappa){
		if (kappa<0)
			return "Poor";
		else if ((kappa>=0)&&(kappa<=0.20))
			return "Slight";
		else if ((kappa>=0.20)&&(kappa<=0.40))
			return "Fair";
		else if ((kappa>0.40)&&(kappa<=0.60))
			return "Moderate";
		else if ((kappa>0.60)&&(kappa<=0.80))
			return "Substantial";
		else if (kappa>0.80)
			return "Almost Perfect";
		else
			return "Not Applicable";
	}
	
	public static String kappaClassificationFleiss(double kappa){
		if (kappa<0)
			return "Poor";
		else if ((kappa>=0)&&(kappa<=0.40))
			return "Marginal";
		else if ((kappa>0.4)&&(kappa<=0.75))
			return "Good";
		else if (kappa>0.75)
			return "Excellent";
		else
			return "Not Applicable";
	}
	
	
}
