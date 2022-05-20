package simulator;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MushDetector {
    private static void MLJ48() {
        try {
            DataSource source = new DataSource("mushroom.arff");
            Instances ds = source.getDataSet();
            ds.setClassIndex(ds.numAttributes()-1);

            J48 classifier = new J48();
            classifier.buildClassifier(ds);

            Visualizer v = new Visualizer();
            v.start(classifier);

        } catch (Exception e) {
            System.err.println("Something went wrong");
        }
    }


	public static void main(String[] args) {
        Simulator sim = new Simulator();	
        FIS fis = FIS.load("system.fcl", true);

        if (fis == null) {
            System.err.println("Can't load fcl file");
            System.exit(1);
        }

        FunctionBlock fb = fis.getFunctionBlock(null);
        while (true) {
        	sim.step();
            // TODO : find a way to test the mushroom on the screen with my algorithm
            System.out.println(sim.getMushroomAttributes());
        }
        

        
        /*
        while(true) {
            sim.step();
        }
        */
	}
}
