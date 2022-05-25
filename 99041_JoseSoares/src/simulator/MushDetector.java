package simulator;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MushDetector {
    private static J48 MLJ48() {
        J48 classifier = null;
        try {
            DataSource source = new DataSource("mushroom.arff");
            Instances dataset = source.getDataSet();
            dataset.setClassIndex(dataset.numAttributes()-1);

            classifier = new J48();
            classifier.buildClassifier(dataset);
        } catch (Exception e) {
            System.err.println("Something went wrong when building classifier");
        }

        return classifier;

    }

    private static String predictMushroom(J48 classifier, String[] mushroom) {
        String pred = "";
        try {
            DataSource source = new DataSource("mushroom.arff");
            Instances dataset = source.getDataSet();
            dataset.setClassIndex(dataset.numAttributes()-1);

            NewInstances ni = new NewInstances(dataset);
            ni.addInstance(mushroom);

           Instances predict_dt = ni.getDataset();

            for (int i = 0; i < predict_dt.numInstances(); i++) {
                Instance inst = predict_dt.instance(i);

                double predict = classifier.classifyInstance(inst);
                pred = predict_dt.classAttribute().value((int) (predict));
            }
        } catch (Exception e) {
            System.err.println("Something went wrong when predicting");
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
        sim.setSimulationSpeed(50);

        J48 classifier = MLJ48();

        while (true) {
            sim.step();
            sim.setAction(Action.NO_ACTION);
            double rightDistance = sim.getDistanceR();
            double leftDistance = sim.getDistanceL();
            double centerDistance = sim.getDistanceC();
            fb.setVariable("sensor_right", rightDistance);
            fb.setVariable("sensor_left", leftDistance);
            fb.setVariable("sensor_center", centerDistance);
            fb.evaluate();
            sim.setRobotAngle(fb.getVariable("wheel_angle").defuzzify());
            
            if (sim.getMushroomAttributes() != null) {
                String mushroomPrediction = predictMushroom(classifier, sim.getMushroomAttributes());
                if (mushroomPrediction.equals("poisonous")) 
                    fb.setVariable("mushroom", 25);
                else if (mushroomPrediction.equals("edible"))
                    fb.setVariable("mushroom", 15);
                else 
                    fb.setVariable("mushroom", 5);
                
                fb.evaluate();

                sim.setRobotAngle(fb.getVariable("wheel_angle").defuzzify());
                double action = fb.getVariable("action").defuzzify();
                 
                if (Math.min(leftDistance, Math.min(centerDistance, rightDistance)) < 1) {
                    if (action == 0.0) sim.setAction(Action.NO_ACTION);
                    else if (action == 1.0) sim.setAction(Action.DESTROY);
                    else if (action == 2.0) sim.setAction(Action.PICK_UP);
   
                }
            }

        }
    }

}
