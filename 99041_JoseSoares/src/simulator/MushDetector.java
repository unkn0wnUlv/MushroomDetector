package simulator;

import javax.swing.ActionMap;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import weka.classifiers.trees.J48;
import simulator.NewInstances;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MushDetector {
    private static String MLJ48(String[] mushroom) {
        String pred = "";
        try {
            DataSource source = new DataSource("mushroom.arff");
            Instances dataset = source.getDataSet();
            dataset.setClassIndex(dataset.numAttributes()-1);

            J48 classifier = new J48();
            classifier.buildClassifier(dataset);
            
            /*
            Visualizer v = new Visualizer();
            v.start(classifier);
            */

            // The values bellow are for test porpuses only
			NewInstances ni = new NewInstances(dataset);
			ni.addInstance(mushroom);

			Instances predict_dt = ni.getDataset();

			for (int i = 0; i < predict_dt.numInstances(); i++) {
				Instance inst = predict_dt.instance(i);
                
				String act = inst.stringValue(inst.numAttributes() - 1);
				double actual = inst.classValue();

				double predict = classifier.classifyInstance(inst);
				pred = predict_dt.classAttribute().value((int) (predict));
			}

        } catch (Exception e) {
            System.err.println("Something went wrong");
        }

        return pred;
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
            double rightDistance = sim.getDistanceR();
            double leftDistance = sim.getDistanceL();
            double centerDistance = sim.getDistanceC();
            fb.setVariable("sensor_right", rightDistance);
            fb.setVariable("sensor_left", leftDistance);
            fb.setVariable("sensor_center", centerDistance);
            fb.evaluate();
            sim.setRobotAngle(fb.getVariable("wheel_angle").defuzzify());
            
            if (sim.getMushroomAttributes() != null) {
                String mushroomPrediction = MLJ48(sim.getMushroomAttributes());
                if (mushroomPrediction.equals("poisonous")) 
                    fb.setVariable("mushroom", 25);
                else if (mushroomPrediction.equals("edible"))
                    fb.setVariable("mushroom", 15);
                else 
                    fb.setVariable("mushroom", 5);
                
                fb.evaluate();

                sim.setRobotAngle(fb.getVariable("wheel_angle").defuzzify());
                double action = fb.getVariable("action").defuzzify();

                if (action == 0.0) sim.setAction(Action.NO_ACTION);
                else if (action == 1.0) sim.setAction(Action.DESTROY);
                else if (action == 2.0) sim.setAction(Action.PICK_UP);
                sim.step();
                break;     
            }

        }
    }

}
