package it.cnr.datamining.riskassessment.data.interpretation;

public class StandardDataInterpreter extends ClusterInterpreter{

	public String interpretCluster(String[] cluster_quantised, String[] headers_clust_stat_elements) {
		int riskLevelH = 0;
		int riskLevelM = 0;
		int riskLevelL = 0;

		for (int i = 0; i < cluster_quantised.length; i++) {

			String featurename = headers_clust_stat_elements[i];
			if (featurename.contains("clusterid")) {
				System.out.println("Interpreting cluster " + cluster_quantised[i]);
				continue;
			}
			String quantisation = cluster_quantised[i];
			String risk = getrisk(featurename, quantisation);

			if (risk == "High") {
				System.out.println("\t" + featurename + "->Quart:" + quantisation + "->Risk:" + risk);
				riskLevelH++;
			}
			if (risk == "Medium") {

				riskLevelM++;
			}
			if (risk == "Low") {
				System.out.println("\t" + featurename + "->Quart:" + quantisation + "->Risk:" + risk);
				riskLevelL++;
			}

		}

		int total = riskLevelH + riskLevelL + riskLevelM;
		double highRiskLevel = (double) riskLevelH / (double) total;
		double mediumRiskLevel = (double) riskLevelM / (double) total;
		double lowRiskLevel = (double) riskLevelL / (double) total;
		System.out.println("Levels: H:" + highRiskLevel + " L:" + lowRiskLevel + " M:" + mediumRiskLevel);

		if (highRiskLevel > 0.6) {
			System.out.println("#Very High risk!");
			return "Very High";
		} else if (highRiskLevel > 0.3) {
			System.out.println("#High risk!");
			return "High";
		} else if (lowRiskLevel > 0.5) {
			System.out.println("#Very Low risk!");
			return "Very Low";
		} else if (lowRiskLevel > 0.4) {
			System.out.println("#Low risk!");
			return "Low";
		} else {
			System.out.println("#Moderate risk!");
			return "Moderate";
		}
	}

	public String getrisk(String featurename, String quantisation) {

		String risk = quantisation;
		
		return risk;
	}



}
