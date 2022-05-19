package simulator;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.trees.J48;

public class MLJ48 {
	public static void main(String[] args) {
		try {
			DataSource source = new DataSource("mushroom.arff");
			Instances dataset = source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes() - 1);
			J48 classifier = new J48();
			classifier.buildClassifier(dataset);
			
			// Visualize decision tree
			//Visualizer v = new Visualizer();
			//v.start(classifier);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
